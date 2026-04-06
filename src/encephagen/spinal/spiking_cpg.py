"""Spiking Spinal CPG with Identified Interneuron Classes.

Based on the mammalian locomotor CPG architecture from:
  Rybak et al. (2015) J Physiol
  Danner et al. (2017) eLife
  Kiehn (2016) Nat Rev Neurosci

Interneuron classes:
  Shox2 (excitatory) — rhythm generator, pacemaker-like
  V1 (inhibitory) — ipsilateral flexor-extensor alternation
  V2a (excitatory) — drives commissural V0v
  V2b (inhibitory) — ipsilateral flexor-extensor alternation
  V0d (inhibitory) — commissural, left-right alternation (walk)
  V0v (excitatory) — commissural, left-right alternation (trot)
  V3 (excitatory) — commissural, left-right symmetry

Architecture per hemicord:
  Rhythm Generator (RG): Shox2 flexor ↔ Shox2 extensor (half-center)
  Pattern Formation (PF): V1/V2b for flexor-extensor alternation
  Motor Neurons: Flexor MN, Extensor MN

Commissural connections:
  V0d: inhibits contralateral flexor (walk alternation)
  V2a→V0v: excites contralateral inhibition (trot alternation)
  V3: excites contralateral broadly (symmetry)

This circuit should produce alternating left-right, flexor-extensor gait
purely from its wiring — like the fly's grooming circuit.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SpikingCPG(nn.Module):
    """Spiking locomotor CPG with biologically identified interneuron classes.

    ~80 LIF neurons with specific connectivity.
    Produces quadrupedal alternating gait from circuit structure alone.
    """

    def __init__(self, dt: float = 0.1, device: str = "cuda"):
        super().__init__()
        self.dt = dt
        self.device = torch.device(device)

        # Neuron counts per hemicord (left and right)
        # Total: ~80 neurons
        self.n_per_side = {
            'flex_rg': 5,    # Shox2 flexor rhythm generator (pacemaker)
            'ext_rg': 5,     # Shox2 extensor rhythm generator
            'v1': 4,         # Ipsilateral inhibitory (flex-ext alternation)
            'v2a': 3,        # Ipsilateral excitatory (drives V0v)
            'v2b': 4,        # Ipsilateral inhibitory (flex-ext alternation)
            'v0d': 3,        # Commissural inhibitory (L-R alternation)
            'v0v': 3,        # Commissural excitatory
            'v3': 3,         # Commissural excitatory (symmetry)
            'flex_mn': 5,    # Flexor motor neurons
            'ext_mn': 5,     # Extensor motor neurons
        }
        self.n_side = sum(self.n_per_side.values())  # 40 per side
        self.n_total = self.n_side * 2  # 80 total

        # Build neuron index map
        self.idx = {}  # (side, type) → slice
        offset = 0
        for side in ['L', 'R']:
            for ntype, count in self.n_per_side.items():
                key = f"{side}_{ntype}"
                self.idx[key] = slice(offset, offset + count)
                offset += count

        # Neuron parameters
        # Pacemaker neurons (flex_rg) have persistent Na+ current → burst capability
        tau_m = torch.full((self.n_total,), 20.0)
        v_threshold = torch.full((self.n_total,), 20.0)

        # ALL CPG neurons get lower threshold — these are active circuits
        for side in ['L', 'R']:
            for ntype in self.n_per_side:
                tau_m[self.idx[f'{side}_{ntype}']] = 15.0
                v_threshold[self.idx[f'{side}_{ntype}']] = 8.0
            # Flex RG is the fastest (pacemaker)
            tau_m[self.idx[f'{side}_flex_rg']] = 12.0

        self.register_buffer('tau_m', tau_m)
        self.register_buffer('v_threshold', v_threshold)

        # Build connectivity matrix (the circuit that produces behavior)
        W = self._build_circuit()
        self.register_buffer('W', W)

        # Background tonic drive (like descending brainstem drive)
        self.tonic_drive = torch.zeros(self.n_total, device=self.device)
        # Tonic drive — calibrated so neurons fire
        for side in ['L', 'R']:
            self.tonic_drive[self.idx[f'{side}_flex_rg']] = 15.0  # well above threshold
            self.tonic_drive[self.idx[f'{side}_ext_rg']] = 13.0
            self.tonic_drive[self.idx[f'{side}_flex_mn']] = 10.0  # ABOVE threshold → tonic firing
            self.tonic_drive[self.idx[f'{side}_ext_mn']] = 10.0  # V1/V2b inhibition creates rhythm
            self.tonic_drive[self.idx[f'{side}_v1']] = 5.0
            self.tonic_drive[self.idx[f'{side}_v2b']] = 5.0
            self.tonic_drive[self.idx[f'{side}_v0d']] = 6.0  # near threshold, RG push fires it

        self.to(self.device)
        print(f"  Spiking CPG: {self.n_total} neurons, {int((W != 0).sum())} synapses")

    def _build_circuit(self):
        """Build the specific connectivity that produces locomotion.

        This is the equivalent of the fly's connectome for grooming —
        every connection is specific and purposeful.
        """
        W = torch.zeros(self.n_total, self.n_total)

        def connect(src, dst, weight, normalize=True):
            """Connect all neurons in src group to all in dst group."""
            n_src = src.stop - src.start
            w = weight / max(n_src, 1) if normalize else weight
            for i in range(src.start, src.stop):
                for j in range(dst.start, dst.stop):
                    if i != j:
                        W[i, j] = w

        for side in ['L', 'R']:
            other = 'R' if side == 'L' else 'L'

            # ============================================
            # RHYTHM GENERATOR: Flexor-Extensor half-center
            # Mutual inhibition produces oscillation
            # ============================================
            flex_rg = self.idx[f'{side}_flex_rg']
            ext_rg = self.idx[f'{side}_ext_rg']

            # Flexor RG recurrent excitation (sustains burst)
            connect(flex_rg, flex_rg, 0.3)
            # Extensor RG recurrent excitation
            connect(ext_rg, ext_rg, 0.3)
            # Mutual inhibition (proven parameters from 2-neuron test)
            connect(flex_rg, ext_rg, -2.0)  # flex inhibits ext
            connect(ext_rg, flex_rg, -1.5)  # ext inhibits flex (weaker → flexor-driven)

            # ============================================
            # PATTERN FORMATION: V1/V2b for flex-ext alternation
            # ============================================
            v1 = self.idx[f'{side}_v1']
            v2b = self.idx[f'{side}_v2b']
            flex_mn = self.idx[f'{side}_flex_mn']
            ext_mn = self.idx[f'{side}_ext_mn']

            # RG drives PF layer (strong — this creates the rhythm in MNs)
            connect(flex_rg, v1, 2.0, normalize=False)
            connect(ext_rg, v2b, 2.0, normalize=False)

            # V1 inhibits extensor MN during flexion (strong)
            connect(v1, ext_mn, -3.0, normalize=False)
            # V2b inhibits flexor MN during extension (strong)
            connect(v2b, flex_mn, -3.0, normalize=False)

            # RG directly excites corresponding MNs (NOT normalized — population drive)
            connect(flex_rg, flex_mn, 2.0, normalize=False)
            connect(ext_rg, ext_mn, 2.0, normalize=False)

            # ============================================
            # COMMISSURAL: Left-Right alternation
            # ============================================
            v0d = self.idx[f'{side}_v0d']
            v2a = self.idx[f'{side}_v2a']
            v0v = self.idx[f'{side}_v0v']
            v3 = self.idx[f'{side}_v3']

            contra_flex_rg = self.idx[f'{other}_flex_rg']
            contra_ext_rg = self.idx[f'{other}_ext_rg']
            contra_flex_mn = self.idx[f'{other}_flex_mn']

            # V0d: inhibits contralateral flexor → L-R alternation (strong)
            connect(flex_rg, v0d, 2.0, normalize=False)
            connect(v0d, contra_flex_rg, -3.0, normalize=False)

            # V2a → V0v: faster gait pathway (weak for now)
            connect(flex_rg, v2a, 0.3)
            connect(v2a, v0v, 0.4)
            connect(v0v, contra_flex_mn, -0.4)

            # V3: symmetry (weak)
            connect(ext_rg, v3, 0.3)
            connect(v3, contra_ext_rg, 0.2)

        return W

    def init_state(self):
        """Initialize neuron state with asymmetric initial conditions."""
        v = torch.rand(self.n_total, device=self.device) * 5.0
        # Break symmetry: left flexor starts higher
        v[self.idx['L_flex_rg']] = 7.0  # near threshold (8.0)
        v[self.idx['R_ext_rg']] = 6.0

        return {
            'v': v,
            'refrac': torch.zeros(self.n_total, device=self.device),
            'i_syn': torch.zeros(self.n_total, device=self.device),
            'adaptation': torch.zeros(self.n_total, device=self.device),
        }

    def step(self, state, drive_modulation=0.0):
        """One timestep of the spiking CPG.

        Args:
            state: neuron state dict
            drive_modulation: external drive (from brainstem), [-1, 1]

        Returns:
            state, motor_output dict
        """
        v = state['v']
        refrac = state['refrac']
        i_syn = state['i_syn']
        adaptation = state['adaptation']

        # Tonic drive + modulation
        drive = self.tonic_drive * (1.0 + drive_modulation * 0.5)

        # Background noise
        noise = torch.randn(self.n_total, device=self.device) * 0.5

        # Total input: drive + synaptic - adaptation + noise
        # Adaptation is the KEY mechanism: active neurons fatigue,
        # releasing the opposing half-center → oscillation
        i_total = i_syn + drive - adaptation + noise

        # Membrane dynamics
        active = refrac <= 0
        dv = (-v + i_total) / self.tau_m
        v = v + self.dt * dv * active.float()

        # Spike detection
        spikes = (v >= self.v_threshold) & active

        # Reset — RG neurons reset to above-zero voltage (burst tendency)
        # This mimics persistent Na+ current that makes RG neurons burst
        reset_v = torch.zeros_like(v)
        for side in ['L', 'R']:
            reset_v[self.idx[f'{side}_flex_rg']] = 4.0  # high reset → next spike comes fast
            reset_v[self.idx[f'{side}_ext_rg']] = 4.0
        v = torch.where(spikes, reset_v, v)
        refrac = torch.where(spikes, torch.full_like(refrac, 1.0), refrac)  # shorter refractory
        refrac = torch.clamp(refrac - self.dt, min=0)

        # Adaptation: increases with each spike, decays slowly
        # This is what makes the half-center oscillate:
        # fire → adapt → fatigue → stop → release other half → it fires → etc
        tau_adapt = 100.0  # slow adaptation (100ms) — controls oscillation period
        beta_adapt = 2.0   # strength of adaptation per spike (proven from 2-neuron test)
        adaptation = adaptation * np.exp(-self.dt / tau_adapt) + spikes.float() * beta_adapt

        # Synaptic current from spikes
        syn_input = self.W @ spikes.float()
        tau_syn = 5.0
        i_syn = i_syn * np.exp(-self.dt / tau_syn) + syn_input

        # Extract motor output (firing rates of motor neurons)
        motor = {}
        for side in ['L', 'R']:
            flex_spikes = spikes[self.idx[f'{side}_flex_mn']].float().mean().item()
            ext_spikes = spikes[self.idx[f'{side}_ext_mn']].float().mean().item()
            motor[f'{side}_flex'] = flex_spikes
            motor[f'{side}_ext'] = ext_spikes
            motor[f'{side}_torque'] = ext_spikes - flex_spikes  # net torque

        new_state = {'v': v, 'refrac': refrac, 'i_syn': i_syn, 'adaptation': adaptation}
        return new_state, motor, spikes

    def get_torques(self, motor, scale=1.0):
        """Convert motor neuron output to joint torques.

        Returns [right_hip, right_knee, left_hip, left_knee]
        """
        r_torque = motor['R_torque'] * scale
        l_torque = motor['L_torque'] * scale
        knee_ratio = 0.7

        return np.array([
            np.clip(r_torque, -1, 1),
            np.clip(r_torque * knee_ratio, -1, 1),
            np.clip(l_torque, -1, 1),
            np.clip(l_torque * knee_ratio, -1, 1),
        ])


def test_cpg():
    """Test: does the spiking CPG produce alternating gait from circuit structure?"""
    cpg = SpikingCPG(device="cuda")
    state = cpg.init_state()

    print("\n  Testing spiking CPG (5000 steps = 500ms)...")
    print(f"  {'Step':>6} {'L_flex':>8} {'L_ext':>8} {'R_flex':>8} {'R_ext':>8} {'L_torq':>8} {'R_torq':>8}")

    # Accumulate over windows
    window = 100
    L_flex_acc, L_ext_acc, R_flex_acc, R_ext_acc = 0, 0, 0, 0

    with torch.no_grad():
        for step in range(5000):
            state, motor, spikes = cpg.step(state)

            L_flex_acc += motor['L_flex']
            L_ext_acc += motor['L_ext']
            R_flex_acc += motor['R_flex']
            R_ext_acc += motor['R_ext']

            if (step + 1) % window == 0:
                lf = L_flex_acc / window
                le = L_ext_acc / window
                rf = R_flex_acc / window
                re = R_ext_acc / window
                lt = le - lf
                rt = re - rf

                bar_l = "█" * int(abs(lt) * 50) if lt > 0 else "░" * int(abs(lt) * 50)
                bar_r = "█" * int(abs(rt) * 50) if rt > 0 else "░" * int(abs(rt) * 50)

                print(f"  {step+1:>6} {lf:>8.3f} {le:>8.3f} {rf:>8.3f} {re:>8.3f} "
                      f"{lt:>+8.3f} {rt:>+8.3f}  L:{bar_l:<10} R:{bar_r:<10}")

                L_flex_acc = L_ext_acc = R_flex_acc = R_ext_acc = 0

    # Check for alternation
    print("\n  Checking for alternation pattern...")
    L_history, R_history = [], []
    with torch.no_grad():
        for step in range(10000):
            state, motor, _ = cpg.step(state)
            if step % 100 == 0:
                L_history.append(motor['L_torque'])
                R_history.append(motor['R_torque'])

    L_arr = np.array(L_history)
    R_arr = np.array(R_history)

    if len(L_arr) > 5:
        corr = np.corrcoef(L_arr, R_arr)[0, 1]
        l_var = np.std(L_arr)
        r_var = np.std(R_arr)
        print(f"  L-R correlation: {corr:.3f} (want -1.0 for alternation)")
        print(f"  L variability: {l_var:.4f}  R variability: {r_var:.4f}")
        if corr < -0.3 and l_var > 0.01:
            print("  ✓ ALTERNATING GAIT DETECTED — circuit structure produces locomotion!")
        elif l_var > 0.01:
            print(f"  ~ Oscillation present but not cleanly alternating (corr={corr:.2f})")
        else:
            print("  ✗ No oscillation — circuit needs tuning")


if __name__ == "__main__":
    test_cpg()
