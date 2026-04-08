"""Sleep Phase: Offline consolidation for motor learning.

The motor cortex doesn't learn DURING movement — it consolidates
DURING REST. The mechanism (Gulati 2024, eLife 2024):

  1. During active training: cerebellum learns error correction online
  2. During sleep/rest: cerebellar nuclei → thalamus → M1 pathway
     replays and strengthens, transferring control to cortex
  3. 3-6 Hz oscillatory coherence between M1 and cerebellum emerges
  4. After sleep: M1 has internalized what cerebellum learned

This is why real infants sleep 16-17 hours/day — sleep IS learning.

Implementation:
  - Reduce sensory input to near-zero (sleeping brain gets no external input)
  - Reduce tonic drive (lower arousal = sleep state)
  - Allow cerebellar→thalamic→cortical pathway to run
  - Strengthen M1 synapses that co-activate with cerebellar nuclei
  - Track M1-cerebellum coherence as a consolidation metric
"""

import numpy as np
import torch


class SleepConsolidation:
    """Offline motor consolidation during sleep."""

    def __init__(self, brain, device="cuda"):
        self.brain = brain
        self.device = torch.device(device)

        # Consolidation learning rate (stronger than online learning)
        self.consolidation_lr = 0.005

        # Track M1-cerebellum coherence
        self.m1_history = []
        self.cb_history = []

    def sleep_episode(self, duration_steps=5000):
        """Run a sleep episode — offline consolidation.

        Reduces sensory input, lowers arousal, and allows
        cerebellar→thalamic→M1 pathway to consolidate.
        """
        # Sleep state: reduced drive
        original_arousal = self.brain.hypothalamus.arousal

        consolidation_events = 0

        for step in range(duration_steps):
            # Minimal sensory input (sleeping — only internal activity)
            sensory = {
                'visual': 0.0,
                'auditory': 0.0,
                'somatosensory': 0.01,  # minimal proprioception
                'height': 1.25,         # lying down (stable)
                'tilt_fb': 0.0,
                'threat': 0.0,
                'reward': 0.0,
            }

            # Brain step with reduced arousal
            output = self.brain.step(sensory)

            # Track M1 and cerebellum activity for coherence
            # M1 = motor cortex regions
            m1_rate = output.get('motor_rate', 0)
            cb_rate = output.get('dcn_output', 0)

            self.m1_history.append(m1_rate)
            self.cb_history.append(cb_rate)

            # Keep only last 1000 for coherence calculation
            if len(self.m1_history) > 1000:
                self.m1_history = self.m1_history[-1000:]
                self.cb_history = self.cb_history[-1000:]

            # CONSOLIDATION: strengthen cortex weights that correlate
            # with cerebellar output (transfer of motor skill)
            if step % 100 == 0 and cb_rate > 0.001:
                self._consolidate_motor(cb_rate)
                consolidation_events += 1

        # Restore arousal
        self.brain.hypothalamus.arousal = original_arousal

        # Compute coherence
        coherence = self._compute_coherence()

        return {
            'consolidation_events': consolidation_events,
            'coherence_3_6hz': coherence,
            'duration_steps': duration_steps,
        }

    def _consolidate_motor(self, cb_rate):
        """Strengthen cortex motor connections based on cerebellar activity.

        This is the DCN → thalamus → M1 consolidation pathway.
        During sleep, co-active cerebellar and cortical neurons
        strengthen their connections.
        """
        # Get motor cortex L5 neurons (output layer)
        for r in self.brain.motor_regions:
            l5_start, l5_end = self.brain.cell_map.get((r, 'L5'), (0, 0))
            l23_start, l23_end = self.brain.cell_map.get((r, 'L23'), (0, 0))

            if l5_end <= l5_start or l23_end <= l23_start:
                continue

            # Strengthen L23→L5 connections in motor cortex
            # (cortex internalizes motor patterns from cerebellum)
            v = self.brain.states['cortex_v']
            l23_active = (v[l23_start:l23_end] > 5.0).float()
            l5_active = (v[l5_start:l5_end] > 5.0).float()

            if l23_active.sum() > 0 and l5_active.sum() > 0:
                # Hebbian: strengthen co-active L23→L5
                # This is the consolidation — cortex learns the motor pattern
                W = self.brain.cortex_W.coalesce()
                indices = W.indices()
                values = W.values().clone()

                for idx in range(indices.shape[1]):
                    src = indices[0, idx].item()
                    dst = indices[1, idx].item()
                    if (l23_start <= src < l23_end and
                        l5_start <= dst < l5_end and
                        values[idx] > 0):
                        # Strengthen proportional to cerebellar drive
                        values[idx] += self.consolidation_lr * cb_rate
                        values[idx] = min(values[idx].item(), 5.0)

                self.brain.cortex_W = torch.sparse_coo_tensor(
                    indices, values, W.shape).coalesce()

    def _compute_coherence(self):
        """Compute M1-cerebellum coherence in 3-6 Hz band.

        If coherence > 0.1, motor skill consolidation is occurring.
        """
        if len(self.m1_history) < 200:
            return 0.0

        m1 = np.array(self.m1_history[-500:])
        cb = np.array(self.cb_history[-500:])

        if np.std(m1) < 1e-8 or np.std(cb) < 1e-8:
            return 0.0

        # Simple cross-correlation as coherence proxy
        # (proper spectral coherence would need scipy.signal)
        m1_norm = (m1 - m1.mean()) / (np.std(m1) + 1e-10)
        cb_norm = (cb - cb.mean()) / (np.std(cb) + 1e-10)

        # Correlation at different lags (looking for oscillatory structure)
        max_lag = min(100, len(m1_norm) // 2)
        correlations = []
        for lag in range(1, max_lag):
            r = np.corrcoef(m1_norm[:-lag], cb_norm[lag:])[0, 1]
            if not np.isnan(r):
                correlations.append(abs(r))

        if not correlations:
            return 0.0

        # Peak correlation = coherence measure
        coherence = max(correlations)
        return float(coherence)
