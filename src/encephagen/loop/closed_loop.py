"""ClosedLoopRunner: wire brain + environment into a continuous loop.

    Environment.observe() → sensory encoder → Brain sensory regions
                                                ↓ (message passing)
    Environment.step(action) ← motor decoder ← Brain motor regions
                                                ↑
                                            reward → STDP modulation
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from encephagen.connectome.loader import Connectome
from encephagen.network.spiking_brain import SpikingBrain
from encephagen.neurons.lif import LIFParams
from encephagen.sensory.visual import VisualEncoder, VisualParams
from encephagen.motor.decoder import MotorDecoder, MotorParams
from encephagen.environment.grid_world import GridWorld, GridWorldParams
from encephagen.learning.stdp import STDPRule, STDPParams
from encephagen.learning.homeostatic import HomeostaticPlasticity, HomeostaticParams
from encephagen.analysis.functional_roles import _classify_tvb76_regions


@dataclass
class EpisodeLog:
    """Logged data from one episode."""

    steps: int = 0
    total_reward: float = 0.0
    target_reached: bool = False
    distances: list[float] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)


class ClosedLoopRunner:
    """Wires brain, sensory encoders, motor decoder, and environment together.

    The brain perceives the world through sensory regions,
    acts through motor regions, and learns through STDP
    modulated by reward.
    """

    def __init__(
        self,
        connectome: Connectome,
        neurons_per_region: int = 200,
        global_coupling: float = 0.05,
        ext_rate: float = 3.5,
        enable_learning: bool = True,
        sensory_gain: float = 10.0,
        action_every_ms: float = 50.0,
        stdp_every: int = 10,
        lif_params: LIFParams | None = None,
        seed: int | None = None,
    ):
        self.dt = 0.1  # ms
        self.action_every_steps = max(1, int(action_every_ms / self.dt))
        self.stdp_every = stdp_every
        self.enable_learning = enable_learning

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

        # Region indices
        self.sensory_idx = groups.get("sensory", [])[:4]  # Use 4 sensory regions for 4 directions
        self.motor_idx = groups.get("motor", list(range(connectome.num_regions - 2, connectome.num_regions)))

        if len(self.sensory_idx) < 4:
            # Pad with other regions if not enough sensory
            self.sensory_idx = list(range(4))
        if len(self.motor_idx) < 1:
            self.motor_idx = [connectome.num_regions - 1]

        # Sensory: encode 4 direction values as current to 4 sensory regions
        self.sensory_gain = sensory_gain

        # Motor decoder on the first motor region
        self.motor_decoder = MotorDecoder(
            n_neurons=neurons_per_region,
            params=MotorParams(n_actions=4, window_ms=action_every_ms, noise_sigma=0.05),
            seed=seed,
        )

        # Environment
        self.env = GridWorld(
            params=GridWorldParams(size=10.0, step_size=0.3, target_radius=0.5, max_steps=500),
            seed=seed,
        )

        # Learning (STDP per region + homeostatic)
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

    def _encode_observation(self, obs: np.ndarray) -> dict[int, np.ndarray]:
        """Convert 4-value observation to per-region external currents.

        Each observation value (up/down/left/right) drives one sensory region.
        """
        ext = {}
        for i, reg_idx in enumerate(self.sensory_idx[:4]):
            n = self.brain.neurons_per_region
            # Observation value → current to all neurons in that region
            current = np.full(n, obs[i] * self.sensory_gain, dtype=np.float64)
            ext[reg_idx] = current
        return ext

    def _read_motor(self) -> int:
        """Read action from motor region."""
        if self.motor_idx:
            motor_pop = self.brain.regions[self.motor_idx[0]]
            self.motor_decoder.update(motor_pop.neurons.spikes, self.dt)
        return self.motor_decoder.decode_action()

    def _apply_learning(self, reward: float) -> None:
        """Apply STDP with reward modulation."""
        if not self.enable_learning:
            return

        # Reward modulates STDP amplitude
        # Positive reward → amplify recent LTP
        # Negative reward → amplify recent LTD
        reward_factor = 1.0 + np.clip(reward * 2.0, -0.5, 0.5)

        for i, pop in enumerate(self.brain.regions):
            exc_spikes = pop.neurons.spikes[:pop.n_exc]
            all_spikes = pop.neurons.spikes

            # Temporarily scale STDP by reward
            orig_a_plus = self.stdp_rules[i].p.a_plus
            orig_a_minus = self.stdp_rules[i].p.a_minus
            self.stdp_rules[i].p.a_plus = orig_a_plus * reward_factor
            self.stdp_rules[i].p.a_minus = orig_a_minus / reward_factor

            pop.exc_conn = self.stdp_rules[i].step(
                self.dt * self.stdp_every, exc_spikes, all_spikes, pop.exc_conn,
            )

            # Restore
            self.stdp_rules[i].p.a_plus = orig_a_plus
            self.stdp_rules[i].p.a_minus = orig_a_minus

            # Homeostatic
            sim_time = self._step_count * self.dt
            pop.exc_conn = self.homeo_rules[i].step(
                self.dt * self.stdp_every, all_spikes, pop.exc_conn,
                apply_every_ms=200.0, current_time=sim_time,
            )

    def run_episode(self) -> EpisodeLog:
        """Run one episode: brain controls agent until done."""
        obs = self.env.reset()
        self.motor_decoder.reset()
        log = EpisodeLog()

        while not self.env.done:
            # Encode sensory observation
            ext_currents = self._encode_observation(obs)

            # Run brain for action_every_steps timesteps
            accumulated_reward = 0.0
            for step in range(self.action_every_steps):
                self.brain.step(self.dt, external_currents=ext_currents)
                self._step_count += 1

                # Update motor decoder
                if self.motor_idx:
                    motor_pop = self.brain.regions[self.motor_idx[0]]
                    self.motor_decoder.update(motor_pop.neurons.spikes, self.dt)

                # STDP learning (every N steps)
                if self.enable_learning and step % self.stdp_every == 0:
                    self._apply_learning(accumulated_reward)

            # Decode action from motor region
            action = self.motor_decoder.decode_action()

            # Step environment
            obs, reward, done = self.env.step(action)
            accumulated_reward += reward

            # Log
            log.steps += 1
            log.total_reward += reward
            log.distances.append(self.env.distance_to_target)
            log.actions.append(action)

        log.target_reached = self.env.target_reached
        return log

    def run_episodes(self, n_episodes: int, log_every: int = 10) -> list[EpisodeLog]:
        """Run multiple episodes, tracking progress."""
        logs = []
        t0 = time.time()

        for ep in range(n_episodes):
            log = self.run_episode()
            logs.append(log)

            if (ep + 1) % log_every == 0:
                recent = logs[-log_every:]
                avg_reward = np.mean([l.total_reward for l in recent])
                avg_steps = np.mean([l.steps for l in recent])
                reach_rate = np.mean([l.target_reached for l in recent])
                elapsed = time.time() - t0
                print(f"  Ep {ep+1:>4}/{n_episodes}  "
                      f"reward={avg_reward:>+6.1f}  "
                      f"steps={avg_steps:>5.0f}  "
                      f"reach={reach_rate:>4.0%}  "
                      f"({elapsed:.0f}s)")

        return logs
