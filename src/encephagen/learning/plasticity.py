"""Distributed Plasticity: 7 learning rules for 7 brain organs.

Each organ learns differently, simultaneously:
  Cerebellum:  Supervised LTD (climbing fiber error → weaken parallel fiber synapses)
  Basal ganglia: Reinforcement (dopamine RPE → corticostriatal plasticity)
  Hippocampus: Hebbian LTP (co-active CA3 neurons strengthen connections)
  Amygdala:    Pavlovian (threat + sensory → LA-CeA strengthening)
  Cortex:      Homeostatic (firing rate adaptation — slow, statistical)
  Brainstem:   Habituation (repeated stimulus → synaptic depression)
  CPG:         Sensory adaptation (proprioceptive feedback adjusts timing)

All rules are LOCAL — each synapse only uses information available
at that synapse (pre, post, neuromodulator). No backprop.
"""

import numpy as np
import torch


class DistributedPlasticity:
    """Manages learning across all brain organs simultaneously."""

    def __init__(self, brain, device="cuda"):
        self.brain = brain
        self.device = torch.device(device)
        self.step_count = 0

        # Learning rates per organ
        self.lr = {
            'cerebellum': 0.001,     # supervised: moderate
            'basal_ganglia': 0.002,  # RL: moderate
            'hippocampus': 0.005,    # Hebbian: fast (one-shot)
            'amygdala': 0.005,       # Pavlovian: fast (one-trial)
            'cortex': 0.0001,        # homeostatic: very slow
            'brainstem': 0.01,       # habituation: fast
            'cpg': 0.0005,           # adaptation: slow
        }

        # Brainstem habituation tracking
        self.brainstem_stim_history = {}

        # Weight bounds
        self.w_max = 10.0
        self.w_min = -10.0

        print("  Distributed plasticity: 7 learning rules enabled")

    def step(self, brain_output, sensory_input):
        """Apply all learning rules for one timestep.

        Called AFTER brain.step() with the output and input.
        """
        self.step_count += 1

        # Only apply plasticity every N steps (for efficiency)
        if self.step_count % 10 != 0:
            return

        reward = sensory_input.get('reward', 0)
        threat = sensory_input.get('threat', 0)
        error = abs(sensory_input.get('tilt_fb', 0)) * 2  # balance error

        # 1. Cerebellum: supervised LTD
        self._cerebellum_learn(error)

        # 2. Basal ganglia: dopamine RL
        self._bg_learn(reward)

        # 3. Hippocampus: Hebbian LTP
        self._hippocampus_learn()

        # 4. Amygdala: Pavlovian
        self._amygdala_learn(threat)

        # 5. Brainstem: habituation
        self._brainstem_habituate(sensory_input)

        # 6. CPG: sensory adaptation
        self._cpg_adapt(sensory_input)

    def _cerebellum_learn(self, error):
        """Supervised LTD at parallel fiber → Purkinje synapses.

        When climbing fiber (error) is active AND parallel fiber (granule)
        fired recently → WEAKEN that synapse. This teaches the cerebellum
        to predict better and reduce errors.

        dW = -lr * error * pre_activity * post_activity
        """
        if error < 0.1:
            return

        cb = self.brain.cerebellum
        W = cb.W
        state = self.brain.states['cerebellum']

        # Granule → Purkinje connections
        gr = cb.idx['granule']; pk = cb.idx['purkinje']
        gr_v = state['v'][gr.start:gr.stop]
        pk_v = state['v'][pk.start:pk.stop]

        # Pre active (granule near threshold) AND post active (purkinje) AND error
        gr_active = (gr_v > 5.0).float()
        pk_active = (pk_v > 5.0).float()

        if gr_active.sum() > 0 and pk_active.sum() > 0:
            # LTD: weaken active granule→purkinje connections
            lr = self.lr['cerebellum'] * error
            for i in range(gr.start, gr.stop):
                if gr_v[i - gr.start] > 5.0:
                    for j in range(pk.start, pk.stop):
                        if pk_v[j - pk.start] > 5.0 and W[i, j] > 0:
                            W[i, j] = max(W[i, j] - lr, 0.01)

    def _bg_learn(self, reward):
        """Dopamine-modulated reinforcement learning at corticostriatal synapses.

        Dopamine = reward prediction error (RPE):
          RPE > 0: "better than expected" → strengthen D1 (GO), weaken D2 (STOP)
          RPE < 0: "worse than expected" → weaken D1, strengthen D2

        This is the core of habit learning.
        """
        bg = self.brain.bg
        rpe = reward - bg.dopamine  # reward prediction error

        if abs(rpe) < 0.05:
            return

        lr = self.lr['basal_ganglia'] * abs(rpe)

        # D1 MSNs: RPE > 0 → strengthen inputs (GO more)
        # D2 MSNs: RPE < 0 → strengthen inputs (STOP more)
        # Simplified: adjust tonic drive of D1/D2 based on RPE
        if rpe > 0:
            # Reward! Increase D1 excitability → easier to GO
            for a in range(bg.n_actions):
                d1 = bg.action_idx[a]['d1']
                # Strengthen recently active D1 channels
                bg.tonic_drive[d1.start:d1.stop] += lr * 0.5
        else:
            # Punishment! Increase D2 → easier to STOP
            for a in range(bg.n_actions):
                d2 = bg.action_idx[a]['d2']
                bg.tonic_drive[d2.start:d2.stop] += lr * 0.3

        # Clamp
        bg.tonic_drive = torch.clamp(bg.tonic_drive, 0, 15)

    def _hippocampus_learn(self):
        """Hebbian LTP in CA3 recurrent connections.

        "Neurons that fire together, wire together."
        CA3 neurons that co-activate strengthen their connections.
        This is pattern completion — the basis of memory.
        """
        hp = self.brain.hippocampus
        state = self.brain.states['hippocampus']
        ca3 = hp.idx['ca3']
        ca3_v = state['v'][ca3.start:ca3.stop]

        # Find active CA3 neurons
        active = (ca3_v > 5.0).float()
        n_active = active.sum().item()

        if n_active < 2 or n_active > 20:  # need some but not too many
            return

        # Hebbian: strengthen connections between co-active neurons
        lr = self.lr['hippocampus']
        W = hp.W
        for i in range(ca3.start, ca3.stop):
            if ca3_v[i - ca3.start] > 5.0:
                for j in range(ca3.start, ca3.stop):
                    if i != j and ca3_v[j - ca3.start] > 5.0:
                        if W[i, j] > 0:  # only excitatory
                            W[i, j] = min(W[i, j] + lr, self.w_max)

    def _amygdala_learn(self, threat):
        """Pavlovian fear conditioning at LA → CeA synapses.

        Threat stimulus + sensory context → strengthen LA→BLA→CeA pathway.
        One-trial learning: a single scary event creates a lasting association.
        """
        if threat < 0.3:
            return

        am = self.brain.amygdala
        state = self.brain.states['amygdala']
        la = am.idx['la']
        cea = am.idx['cea']

        la_active = (state['v'][la.start:la.stop] > 5.0).float()
        if la_active.sum() < 1:
            return

        # Strengthen LA → CeA connections (direct fear pathway)
        lr = self.lr['amygdala'] * threat
        W = am.W
        for i in range(la.start, la.stop):
            if state['v'][i] > 5.0:
                for j in range(cea.start, cea.stop):
                    if W[i, j] > 0:
                        W[i, j] = min(W[i, j] + lr, self.w_max)

    def _brainstem_habituate(self, sensory_input):
        """Habituation: repeated stimulus → decreased response.

        If the same stimulus keeps happening and nothing bad follows,
        stop reacting to it. Simplest form of learning.
        """
        # Track recent stimuli
        for key in ['visual', 'auditory', 'somatosensory']:
            val = sensory_input.get(key, 0)
            if key not in self.brainstem_stim_history:
                self.brainstem_stim_history[key] = []
            self.brainstem_stim_history[key].append(val)
            # Keep last 100
            if len(self.brainstem_stim_history[key]) > 100:
                self.brainstem_stim_history[key] = self.brainstem_stim_history[key][-100:]

        # If a stimulus has been constant for 50+ steps → habituate
        for key, history in self.brainstem_stim_history.items():
            if len(history) >= 50:
                recent = history[-50:]
                if np.std(recent) < 0.05 and np.mean(recent) > 0.1:
                    # Constant stimulus → reduce brainstem sensitivity
                    # (This happens through the startle threshold)
                    self.brain.brainstem.startle_level *= 0.99

    def _cpg_adapt(self, sensory_input):
        """Sensory-driven CPG adaptation.

        Proprioceptive feedback adjusts CPG tonic drive.
        If the body is unstable → increase CPG drive (step more).
        If stable → relax CPG drive.
        """
        tilt = abs(sensory_input.get('tilt_fb', 0))
        height = sensory_input.get('height', 1.25)

        cpg = self.brain.cpg
        lr = self.lr['cpg']

        # Unstable (high tilt) → increase CPG flex drive
        if tilt > 0.3:
            for s in ['L', 'R']:
                idx = cpg.idx[f'{s}_flex_rg']
                cpg.tonic_drive[idx.start:idx.stop] += lr
                cpg.tonic_drive[idx.start:idx.stop] = torch.clamp(
                    cpg.tonic_drive[idx.start:idx.stop], 0, 25)

        # Stable → slowly relax toward baseline
        elif tilt < 0.1:
            for s in ['L', 'R']:
                idx = cpg.idx[f'{s}_flex_rg']
                cpg.tonic_drive[idx.start:idx.stop] *= (1.0 - lr * 0.1)

    def get_learning_stats(self):
        """Return summary of what's been learned."""
        return {
            'total_steps': self.step_count,
            'plasticity_events': self.step_count // 10,
            'bg_dopamine': self.brain.bg.dopamine,
            'amygdala_fear': self.brain.amygdala.fear_level,
            'habituation_active': any(
                len(h) >= 50 and np.std(h[-50:]) < 0.05
                for h in self.brainstem_stim_history.values()
            ),
        }
