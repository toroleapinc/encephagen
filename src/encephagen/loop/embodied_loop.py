"""EmbodiedLoopRunner: brain controls a MuJoCo body in a physics world.

    Body.get_sensory_input() → encode as current → Brain sensory regions
                                                     ↓ (connectome)
    Body.step(torques) ← decode motor output ← Brain motor regions
                                                     ↑
                                                 reward → STDP
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from encephagen.connectome.loader import Connectome
from encephagen.network.spiking_brain import SpikingBrain
from encephagen.neurons.lif import LIFParams
from encephagen.motor.decoder import MotorDecoder, MotorParams
from encephagen.body.simple_body import SimpleBody
from encephagen.learning.stdp import STDPRule, STDPParams
from encephagen.learning.homeostatic import HomeostaticPlasticity, HomeostaticParams
from encephagen.analysis.functional_roles import _classify_tvb76_regions


@dataclass
class EmbodiedEpisodeLog:
    """Logged data from one embodied episode."""

    steps: int = 0
    total_reward: float = 0.0
    max_height: float = 0.0
    max_forward: float = 0.0
    fell: bool = False
    heights: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)


class EmbodiedLoopRunner:
    """Wire brain to MuJoCo body in a continuous closed loop.

    Sensory mapping:
      Body 12-dim observation → distributed across sensory regions
      - Joint angles (4) → somatosensory regions (S1, S2)
      - Joint velocities (4) → somatosensory regions
      - Torso state (4) → additional sensory regions

    Motor mapping:
      Motor region firing rates → 4 joint torques
      - 4 neuron groups in motor region → right_hip, right_knee, left_hip, left_knee
    """

    def __init__(
        self,
        connectome: Connectome,
        neurons_per_region: int = 200,
        global_coupling: float = 0.05,
        ext_rate: float = 3.5,
        enable_learning: bool = True,
        sensory_gain: float = 8.0,
        brain_steps_per_action: int = 500,
        physics_steps_per_action: int = 10,
        stdp_every: int = 20,
        lif_params: LIFParams | None = None,
        seed: int | None = None,
    ):
        """
        Args:
            brain_steps_per_action: Brain timesteps (dt=0.1ms) per motor action.
                500 steps = 50ms brain time per action.
            physics_steps_per_action: MuJoCo steps per action.
                10 steps × 0.005s = 50ms physics time per action.
        """
        self.dt = 0.1  # Brain timestep (ms)
        self.brain_steps_per_action = brain_steps_per_action
        self.physics_steps_per_action = physics_steps_per_action
        self.stdp_every = stdp_every
        self.enable_learning = enable_learning
        self.sensory_gain = sensory_gain

        lif_params = lif_params or LIFParams(j_exc=2.0, g_inh=5.0)
        groups = _classify_tvb76_regions(connectome.labels)

        # Brain
        self.brain = SpikingBrain(
            connectome,
            neurons_per_region=neurons_per_region,
            between_conn_prob=0.02,
            global_coupling=global_coupling,
            ext_rate=ext_rate,
            params=lif_params,
            seed=seed,
        )

        # Region mapping
        sensory_all = groups.get("sensory", [])
        motor_all = groups.get("motor", [])

        # Use up to 12 sensory regions for 12-dim body observation
        self.sensory_idx = sensory_all[:12] if len(sensory_all) >= 12 else sensory_all
        # Pad with other regions if needed
        while len(self.sensory_idx) < 12:
            for idx in range(connectome.num_regions):
                if idx not in self.sensory_idx:
                    self.sensory_idx.append(idx)
                    if len(self.sensory_idx) >= 12:
                        break

        # Motor region: use first motor region, decode 4 torques
        self.motor_idx = motor_all[0] if motor_all else connectome.num_regions - 1

        # Motor decoder: 4 continuous outputs for joint torques
        self.motor_decoder = MotorDecoder(
            n_neurons=neurons_per_region,
            params=MotorParams(n_actions=4, window_ms=50.0, noise_sigma=0.05),
            seed=seed,
        )

        # Body
        self.body = SimpleBody()

        # Learning
        if enable_learning:
            self.stdp_rules = {}
            self.homeo_rules = {}
            for i, pop in enumerate(self.brain.regions):
                self.stdp_rules[i] = STDPRule(
                    n_pre=pop.n_exc, n_post=pop.n_neurons,
                    params=STDPParams(a_plus=0.003, a_minus=0.003, w_max=10.0),
                )
                self.homeo_rules[i] = HomeostaticPlasticity(
                    n_neurons=pop.n_neurons,
                    params=HomeostaticParams(target_rate=10.0, tau_homeo=500.0, eta=0.005),
                )

        self._step_count = 0

    def _encode_body_observation(self, obs: np.ndarray) -> dict[int, np.ndarray]:
        """Convert 12-dim body observation to per-region currents.

        Each observation dimension maps to one sensory region.
        Positive values → excitatory current, scaled by gain.
        """
        ext = {}
        n = self.brain.neurons_per_region
        for i, reg_idx in enumerate(self.sensory_idx[:12]):
            if i < len(obs):
                # Observation value → uniform current to all neurons
                val = float(np.clip(obs[i], -1, 1))
                # Map [-1, 1] to [0, gain] — always excitatory, magnitude varies
                current = np.full(n, (val + 1) * 0.5 * self.sensory_gain, dtype=np.float64)
                ext[reg_idx] = current
        return ext

    def _decode_torques(self) -> np.ndarray:
        """Read 4 joint torques from motor region."""
        motor_pop = self.brain.regions[self.motor_idx]
        self.motor_decoder.update(motor_pop.neurons.spikes, self.dt)

        # Get continuous action [0, 1] per group, map to [-1, 1] torques
        rates = self.motor_decoder.get_action_rates()
        max_rate = rates.max()
        if max_rate > 0:
            normalized = rates / max_rate  # [0, 1]
        else:
            normalized = np.zeros(4)

        torques = normalized * 2.0 - 1.0  # Map to [-1, 1]
        return torques

    def _apply_learning(self, reward: float) -> None:
        """Apply reward-modulated STDP."""
        if not self.enable_learning:
            return

        reward_factor = 1.0 + np.clip(reward * 2.0, -0.5, 0.5)
        sim_time = self._step_count * self.dt

        for i, pop in enumerate(self.brain.regions):
            exc_spikes = pop.neurons.spikes[:pop.n_exc]
            all_spikes = pop.neurons.spikes

            orig_plus = self.stdp_rules[i].p.a_plus
            orig_minus = self.stdp_rules[i].p.a_minus
            self.stdp_rules[i].p.a_plus = orig_plus * reward_factor
            self.stdp_rules[i].p.a_minus = orig_minus / reward_factor

            pop.exc_conn = self.stdp_rules[i].step(
                self.dt * self.stdp_every, exc_spikes, all_spikes, pop.exc_conn,
            )

            self.stdp_rules[i].p.a_plus = orig_plus
            self.stdp_rules[i].p.a_minus = orig_minus

            pop.exc_conn = self.homeo_rules[i].step(
                self.dt * self.stdp_every, all_spikes, pop.exc_conn,
                apply_every_ms=200.0, current_time=sim_time,
            )

    def run_episode(self, max_actions: int = 200) -> EmbodiedEpisodeLog:
        """Run one episode: brain controls body until it falls or max_actions."""
        body_state = self.body.reset()
        self.motor_decoder.reset()
        log = EmbodiedEpisodeLog()

        for action_step in range(max_actions):
            # Get body sensory observation
            obs = self.body.get_sensory_input()
            ext_currents = self._encode_body_observation(obs)

            # Run brain for brain_steps_per_action timesteps
            for step in range(self.brain_steps_per_action):
                self.brain.step(self.dt, external_currents=ext_currents)
                self._step_count += 1

                # Update motor decoder every step
                motor_pop = self.brain.regions[self.motor_idx]
                self.motor_decoder.update(motor_pop.neurons.spikes, self.dt)

                # STDP
                if self.enable_learning and step % self.stdp_every == 0:
                    self._apply_learning(log.rewards[-1] if log.rewards else 0.0)

            # Decode torques from motor region
            torques = self._decode_torques()

            # Step physics
            body_state = self.body.step_n(torques, n=self.physics_steps_per_action)
            reward = self.body.compute_reward(body_state)

            # Log
            log.steps += 1
            log.total_reward += reward
            log.heights.append(body_state.torso_height)
            log.rewards.append(reward)
            log.max_height = max(log.max_height, body_state.torso_height)
            log.max_forward = max(log.max_forward, body_state.torso_velocity_x)

            if body_state.is_fallen:
                log.fell = True
                break

        return log

    def run_episodes(self, n_episodes: int, max_actions: int = 200,
                     log_every: int = 5) -> list[EmbodiedEpisodeLog]:
        """Run multiple embodied episodes."""
        logs = []
        t0 = time.time()

        for ep in range(n_episodes):
            log = self.run_episode(max_actions=max_actions)
            logs.append(log)

            if (ep + 1) % log_every == 0:
                recent = logs[-log_every:]
                avg_reward = np.mean([l.total_reward for l in recent])
                avg_steps = np.mean([l.steps for l in recent])
                avg_height = np.mean([np.mean(l.heights) for l in recent])
                fall_rate = np.mean([l.fell for l in recent])
                elapsed = time.time() - t0
                print(f"  Ep {ep+1:>4}/{n_episodes}  "
                      f"reward={avg_reward:>+7.1f}  "
                      f"steps={avg_steps:>5.0f}  "
                      f"height={avg_height:>.2f}  "
                      f"fall={fall_rate:>4.0%}  "
                      f"({elapsed:.0f}s)")

        return logs
