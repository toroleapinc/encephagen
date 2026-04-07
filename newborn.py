"""The Newborn — Biologically correct architecture.

A newborn human's behavior comes from SUBCORTICAL structures:
  - Spinal cord: CPGs (stepping reflex)
  - Brainstem: reflex arcs (Moro, righting, rooting, startle, withdrawal)
  - Basal ganglia: gating (which reflex wins)
  - Cortex: OBSERVER (gradually learns to modulate, then take over)

The cortex is NOT the controller at birth. It becomes the controller
over the first 2-4 months as corticospinal tracts myelinate.

This matches what real neuroscience tells us:
  "The neonatal behavioral repertoire is almost entirely subcortical."

Usage:
  python newborn.py                    # 30s demo
  python newborn.py --render           # Visual (if display available)
  python newborn.py --video            # Save MP4
  python newborn.py --video --humanoid # 3D humanoid body
"""

import argparse
import json
import time

import cv2
import numpy as np
import torch
import gymnasium as gym

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU
from encephagen.spinal.cpg import SpinalCPG, CPGParams
from encephagen.subcortical.brainstem import BrainstemReflexes, BasalGangliaGating
from encephagen.spinal.spiking_cpg import SpikingCPG


class Newborn:
    """A biologically correct newborn human brain + body.

    Architecture:
      Body → Sensory encoding → Brainstem reflexes → Basal ganglia gating
                                                           ↓
                                                      Motor output → Body
                                                           ↑
                                                    Spinal CPG (stepping)
                                                           ↑ (modulation)
                                                    Cortex (16K neurons, observer)
    """

    def __init__(self, body_type="walker2d", device="cuda", use_spiking_cpg=False):
        print("=" * 60)
        print("  THE NEWBORN — Biologically Correct Architecture")
        print("  Brainstem reflexes + Spinal CPG + Cortex observer")
        print("=" * 60)

        self.device = device

        # ---- Subcortical controllers (the ACTUAL controllers at birth) ----
        self.brainstem = BrainstemReflexes()
        self.basal_ganglia = BasalGangliaGating()
        self.use_spiking_cpg = use_spiking_cpg

        if use_spiking_cpg:
            print("  Building SPIKING CPG (80 neurons, identified interneuron classes)...", flush=True)
            self.spiking_cpg = SpikingCPG(device=device)
            # Load CMA-ES optimized weights
            import os
            params_file = "results/best_cpg_params_cmaes.npy"
            if os.path.exists(params_file):
                self._apply_optimized_params(np.load(params_file))
                print("  Loaded CMA-ES optimized weights")
            self.spiking_cpg_state = self.spiking_cpg.init_state()
            # Warmup
            with torch.no_grad():
                for _ in range(3000):
                    self.spiking_cpg_state, _, _ = self.spiking_cpg.step(self.spiking_cpg_state)
            self.cpg = None  # not using Matsuoka
        else:
            self.cpg = SpinalCPG(CPGParams(
                tau=50.0, tau_adapt=500.0, drive=1.0,
                w_mutual=2.5, w_crossed=1.5, beta=2.5,
            ))
            for _ in range(5000):
                self.cpg.step(0.1)
            self.spiking_cpg = None

        # ---- Cortex (observer at birth, learns over time) ----
        print("  Building cortex (16,000 neurons, 80 regions)...", flush=True)
        sc = np.load('src/encephagen/connectome/bundled/neurolib80_weights.npy')
        tl = np.load('src/encephagen/connectome/bundled/neurolib80_tract_lengths.npy')
        labels = json.load(open('src/encephagen/connectome/bundled/neurolib80_labels.json'))
        self.tau_labels = json.load(open(
            'src/encephagen/connectome/bundled/neurolib80_t1t2_labels.json'))
        self.tau_m = np.load('src/encephagen/connectome/bundled/neurolib80_tau_m.npy')
        c = Connectome(sc, labels); c.tract_lengths = tl

        self.npr = 200
        self.n_regions = 80
        self.n_total = self.n_regions * self.npr
        self.device = device

        self.cortex = SpikingBrainGPU(
            connectome=c, neurons_per_region=self.npr,
            internal_conn_prob=0.05, between_conn_prob=0.03,
            global_coupling=5.0, ext_rate_factor=3.5,
            pfc_regions=[], device=device,
            use_delays=True, conduction_velocity=3.5,
            use_t1t2_gradient=True,
        )

        # Cortex region groups for display
        self.cortex_groups = {}
        for key, patterns in [
            ('Somatosens', ['Postcentral', 'Paracentral']),
            ('Motor', ['Precentral', 'Supp_Motor']),
            ('Visual', ['Calcarine', 'Cuneus', 'Occipital']),
            ('Auditory', ['Heschl', 'Temporal_Sup']),
            ('Frontal', ['Frontal_Sup', 'Frontal_Mid']),
            ('Parietal', ['Parietal', 'Angular']),
            ('Temporal', ['Temporal_Mid', 'Fusiform']),
        ]:
            self.cortex_groups[key] = [i for i, l in enumerate(self.tau_labels)
                                        if any(p in l for p in patterns)]

        # Somatosensory input indices
        soma_idx = [i for i, l in enumerate(self.tau_labels)
                    if 'Postcentral' in l or 'Paracentral' in l]
        self.soma_starts = [ri * self.npr for ri in soma_idx]

        # Warmup cortex
        self.cortex_state = self.cortex.init_state(batch_size=1)
        with torch.no_grad():
            for _ in range(2000):
                self.cortex_state, _ = self.cortex.step(self.cortex_state)

        # Cortex modulation strength (starts near zero, increases with "age")
        self.cortex_influence = 0.0  # 0 = newborn (cortex is observer), 1 = 4 months
        self.age_steps = 0

        self.cortex_rates = {}
        self.body_type = body_type
        print("  Ready.\n")

    def _apply_optimized_params(self, params):
        """Apply CMA-ES optimized weights to spiking CPG."""
        (w_mutual_fe, w_mutual_ef, w_rg_mn, w_v0d_in, w_v0d_out,
         w_pf_drive, w_pf_inh, drive_flex, drive_ext, drive_mn,
         drive_v0d, beta_adapt, tau_adapt_ms, reset_v) = params

        cpg = self.spiking_cpg
        W = cpg.W.clone()

        for side in ['L', 'R']:
            other = 'R' if side == 'L' else 'L'
            fr = cpg.idx[f'{side}_flex_rg']; er = cpg.idx[f'{side}_ext_rg']
            fm = cpg.idx[f'{side}_flex_mn']; em = cpg.idx[f'{side}_ext_mn']
            v1 = cpg.idx[f'{side}_v1']; v2b = cpg.idx[f'{side}_v2b']
            v0d = cpg.idx[f'{side}_v0d']; cfr = cpg.idx[f'{other}_flex_rg']

            def set_w(src, dst, w, norm=True):
                n = src.stop - src.start
                wn = w / max(n, 1) if norm else w
                for i in range(src.start, src.stop):
                    for j in range(dst.start, dst.stop):
                        if i != j: W[i, j] = wn

            set_w(fr, er, w_mutual_fe); set_w(er, fr, w_mutual_ef)
            set_w(fr, fm, w_rg_mn, norm=False); set_w(er, em, w_rg_mn, norm=False)
            set_w(fr, v0d, w_v0d_in, norm=False); set_w(v0d, cfr, w_v0d_out, norm=False)
            set_w(fr, v1, w_pf_drive, norm=False); set_w(er, v2b, w_pf_drive, norm=False)
            set_w(v1, em, w_pf_inh, norm=False); set_w(v2b, fm, w_pf_inh, norm=False)

        cpg.W = W
        cpg.tonic_drive = torch.zeros(cpg.n_total, device=self.device)
        for side in ['L', 'R']:
            cpg.tonic_drive[cpg.idx[f'{side}_flex_rg']] = drive_flex
            cpg.tonic_drive[cpg.idx[f'{side}_ext_rg']] = drive_ext
            cpg.tonic_drive[cpg.idx[f'{side}_flex_mn']] = drive_mn
            cpg.tonic_drive[cpg.idx[f'{side}_ext_mn']] = drive_mn
            cpg.tonic_drive[cpg.idx[f'{side}_v1']] = 5.0
            cpg.tonic_drive[cpg.idx[f'{side}_v2b']] = 5.0
            cpg.tonic_drive[cpg.idx[f'{side}_v0d']] = drive_v0d
        # Store adaptation params for custom step
        self._cpg_beta = beta_adapt
        self._cpg_tau = tau_adapt_ms
        self._cpg_reset_v = reset_v

    def _spiking_cpg_step(self, drive_modulation=0.0):
        """Step the spiking CPG with optimized parameters."""
        cpg = self.spiking_cpg
        state = self.spiking_cpg_state
        v = state['v']; refrac = state['refrac']
        i_syn = state['i_syn']; adaptation = state['adaptation']

        drive = cpg.tonic_drive * (1.0 + drive_modulation * 0.5)
        noise = torch.randn(cpg.n_total, device=self.device) * 0.5
        i_total = i_syn + drive - adaptation + noise
        active = refrac <= 0
        dv = (-v + i_total) / cpg.tau_m
        v = v + cpg.dt * dv * active.float()
        spikes = (v >= cpg.v_threshold) & active

        rv = torch.zeros_like(v)
        for s in ['L', 'R']:
            rv[cpg.idx[f'{s}_flex_rg']] = self._cpg_reset_v
            rv[cpg.idx[f'{s}_ext_rg']] = self._cpg_reset_v
        v = torch.where(spikes, rv, v)
        refrac = torch.where(spikes, torch.full_like(refrac, 1.0), refrac)
        refrac = torch.clamp(refrac - cpg.dt, min=0)
        adaptation = adaptation * np.exp(-cpg.dt / self._cpg_tau) + spikes.float() * self._cpg_beta
        syn_input = cpg.W @ spikes.float()
        i_syn = i_syn * np.exp(-cpg.dt / 5.0) + syn_input

        self.spiking_cpg_state = {'v': v, 'refrac': refrac, 'i_syn': i_syn, 'adaptation': adaptation}

        # Motor output
        motor = {}
        for side in ['L', 'R']:
            lf = spikes[cpg.idx[f'{side}_flex_mn']].float().mean().item()
            le = spikes[cpg.idx[f'{side}_ext_mn']].float().mean().item()
            motor[f'{side}_torque'] = le - lf
        return motor

    def extract_sensory(self, obs):
        """Convert body observation to sensory dict for brainstem."""
        if self.body_type == "humanoid":
            return {
                'height': obs[0],
                'tilt_fb': obs[1],
                'tilt_lr': obs[2],
                'angular_vel': obs[2] * 10,  # approximate
                'touch_left': 0.0,
                'touch_right': 0.0,
                'loud_sound': 0.0,
                'face_touch': 0.0,
            }
        else:  # walker2d
            return {
                'height': obs[0],
                'tilt_fb': obs[1],
                'tilt_lr': 0.0,  # 2D, no left-right
                'angular_vel': obs[9] if len(obs) > 9 else 0,
                'touch_left': 0.0,
                'touch_right': 0.0,
                'loud_sound': 0.0,
                'face_touch': 0.0,
            }

    def cortex_step(self, sensory):
        """Feed sensory to cortex and get cortex state (observation, not control)."""
        ext = torch.zeros(1, self.n_total, device=self.device)

        # Feed proprioception to somatosensory cortex
        signal = (abs(sensory['tilt_fb']) * 50.0 +
                  abs(sensory.get('angular_vel', 0)) * 20.0 +
                  max(0, (1.3 - sensory['height'])) * 40.0)

        for s in self.soma_starts:
            ext[0, s:s + self.npr] = signal

        # Run cortex for a few steps
        with torch.no_grad():
            for _ in range(10):
                self.cortex_state, spikes = self.cortex.step(self.cortex_state, ext)

        # Record cortex activity for display
        for gname, gidx in self.cortex_groups.items():
            total = 0
            for ri in gidx:
                total += spikes[0, ri*self.npr:(ri+1)*self.npr].sum().item()
            self.cortex_rates[gname] = total / (len(gidx) * self.npr)

    def step(self, obs):
        """Full newborn step: sense → brainstem reflexes → BG gating → motor output.

        The cortex observes but doesn't control (at birth).
        """
        self.age_steps += 1

        # 1. Extract sensory info
        sensory = self.extract_sensory(obs)

        # 2. Brainstem reflex arcs (the actual controller)
        reflexes = self.brainstem.process(sensory)

        # 3. Basal ganglia gating (prioritize reflexes)
        gated = self.basal_ganglia.gate(reflexes)

        # 4. Spinal CPG (stepping reflex)
        if self.use_spiking_cpg:
            # Run spiking CPG for 20 timesteps per body step
            motor_acc = {'L_torque': 0.0, 'R_torque': 0.0}
            n_cpg_steps = 20
            with torch.no_grad():
                for _ in range(n_cpg_steps):
                    motor = self._spiking_cpg_step(drive_modulation=gated['stepping_drive'])
                    motor_acc['L_torque'] += motor['L_torque']
                    motor_acc['R_torque'] += motor['R_torque']
            # Average and scale
            cpg_torques = np.array([
                motor_acc['R_torque'] / n_cpg_steps * 5.0,  # right hip
                motor_acc['R_torque'] / n_cpg_steps * 3.5,  # right knee
                motor_acc['L_torque'] / n_cpg_steps * 5.0,  # left hip
                motor_acc['L_torque'] / n_cpg_steps * 3.5,  # left knee
            ])
        else:
            cpg_torques = self.cpg.step(2.0, brain_drive=gated['stepping_drive'])

        # 5. Cortex observes (processes sensory, doesn't control)
        self.cortex_step(sensory)

        # 6. Compose motor output
        if self.body_type == "humanoid":
            action = self._compose_humanoid_action(gated, cpg_torques)
        else:
            action = self._compose_walker_action(gated, cpg_torques)

        return action

    def _compose_walker_action(self, gated, cpg_torques):
        """Compose Walker2d action from subcortical commands."""
        action = np.zeros(6, dtype=np.float32)

        # CPG stepping rhythm
        drive = gated['stepping_drive']
        action[0] = cpg_torques[0] * 0.3 * drive + 0.05  # right hip
        action[1] = cpg_torques[1] * 0.2 * drive + 0.1   # right knee
        action[2] = cpg_torques[1] * 0.1 * drive         # right ankle
        action[3] = cpg_torques[2] * 0.3 * drive + 0.05  # left hip
        action[4] = cpg_torques[3] * 0.2 * drive + 0.1   # left knee
        action[5] = cpg_torques[3] * 0.1 * drive         # left ankle

        # Righting reflex (brainstem — highest priority for balance)
        r = gated['righting']
        action[0] += r * 0.3
        action[3] += r * 0.3

        # Moro reflex — arms extend then flex (mapped to hips as we have no arms)
        moro = gated['moro']
        if abs(moro) > 0.1:
            action[0] += moro * 0.2
            action[3] += moro * 0.2

        # Startle — brief motor burst
        startle = gated['startle']
        if startle > 0.1:
            action[0] += startle * 0.15
            action[3] += startle * 0.15
            action[1] += startle * 0.1
            action[4] += startle * 0.1

        # Withdrawal
        if gated['withdrawal_left'] > 0.1:
            action[3] -= gated['withdrawal_left'] * 0.3
            action[4] += gated['withdrawal_left'] * 0.2
        if gated['withdrawal_right'] > 0.1:
            action[0] -= gated['withdrawal_right'] * 0.3
            action[1] += gated['withdrawal_right'] * 0.2

        return np.clip(action, -1.0, 1.0)

    def _compose_humanoid_action(self, gated, cpg_torques):
        """Compose Humanoid action from subcortical commands."""
        action = np.zeros(17, dtype=np.float32)

        # Righting reflex → torso stabilization
        r_fb = gated['righting']
        action[0] = np.clip(r_fb * 0.3, -0.4, 0.4)   # abdomen z
        action[1] = np.clip(r_fb * 0.2, -0.4, 0.4)   # abdomen y

        # CPG → legs
        drive = gated['stepping_drive']
        action[3] = np.clip(cpg_torques[0] * 0.15 * drive + 0.03, -0.4, 0.4)  # right hip
        action[6] = np.clip(cpg_torques[1] * 0.1 * drive + 0.03, -0.4, 0.4)   # right knee
        action[7] = np.clip(cpg_torques[2] * 0.15 * drive + 0.03, -0.4, 0.4)  # left hip
        action[10] = np.clip(cpg_torques[3] * 0.1 * drive + 0.03, -0.4, 0.4)  # left knee

        # Moro reflex → arms extend then flex
        moro = gated['moro']
        if abs(moro) > 0.1:
            action[11] = np.clip(moro * 0.2, -0.4, 0.4)  # right shoulder
            action[14] = np.clip(moro * 0.2, -0.4, 0.4)  # left shoulder
            action[13] = np.clip(-moro * 0.15, -0.4, 0.4) # right elbow
            action[16] = np.clip(-moro * 0.15, -0.4, 0.4) # left elbow

        # Startle → whole body brief burst
        startle = gated['startle']
        if startle > 0.1:
            for a in range(17):
                action[a] += np.clip(startle * 0.1, -0.1, 0.1)

        return np.clip(action, -0.4, 0.4)

    def draw_status_panel(self, width=420, height=480, sensory=None):
        """Draw combined brain + reflex status panel."""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (25, 25, 35)

        # Title
        cv2.putText(panel, "NEWBORN BRAIN", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        cv2.putText(panel, "Brainstem + Spinal CPG + Cortex", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)

        y = 70

        # Brainstem reflexes
        cv2.putText(panel, "BRAINSTEM (controller)", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 100), 1)
        y += 22

        reflexes = {
            'Righting': abs(self.brainstem.prev_tilt) * 3.0,
            'Moro': abs(self.brainstem.moro_timer) / 20.0,
            'Startle': self.brainstem.startle_level,
            'Stepping': 0.3 if self.brainstem.prev_height > 0.8 else 0.0,
        }
        for name, intensity in reflexes.items():
            bar_w = int(min(intensity * 200, 200))
            color = (100, 255, 150) if intensity > 0.1 else (60, 60, 70)
            cv2.putText(panel, name, (15, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1)
            cv2.rectangle(panel, (130, y), (130 + max(bar_w, 1), y + 14), color, -1)
            y += 22

        y += 10
        cv2.line(panel, (10, y), (width - 10, y), (50, 50, 60), 1)
        y += 15

        # Spinal CPG
        cv2.putText(panel, "SPINAL CPG (stepping)", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
        y += 22
        cpg_state = self.cpg.get_state()
        for name in ['left_flex', 'left_ext', 'right_flex', 'right_ext']:
            val = cpg_state[name]
            bar_w = int(min(abs(val) * 100, 150))
            color = (100, 200, 255) if val > 0 else (60, 60, 70)
            cv2.putText(panel, name.replace('_', ' '), (15, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            cv2.rectangle(panel, (130, y), (130 + max(bar_w, 1), y + 12), color, -1)
            y += 18

        y += 10
        cv2.line(panel, (10, y), (width - 10, y), (50, 50, 60), 1)
        y += 15

        # Cortex (observer)
        cv2.putText(panel, "CORTEX (observer — not controlling)", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 100, 255), 1)
        y += 22

        for gname, gidx in self.cortex_groups.items():
            rate = self.cortex_rates.get(gname, 0)
            tau = np.mean([self.tau_m[i] for i in gidx]) if gidx else 0
            bar_w = int(min(rate * 200 * 15, 150))
            cv2.putText(panel, f"{gname}", (15, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 130), 1)
            cv2.rectangle(panel, (130, y), (130 + max(bar_w, 1), y + 12),
                          (120, 80, 200), -1)
            cv2.putText(panel, f"{tau:.0f}ms", (290, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90, 90, 90), 1)
            y += 18

        return panel


def run_demo(body_type="walker2d", duration_s=30, video=False, spiking_cpg=False):
    """Run the biologically correct newborn demo."""
    newborn = Newborn(body_type=body_type, use_spiking_cpg=spiking_cpg)

    env_name = 'Humanoid-v5' if body_type == "humanoid" else 'Walker2d-v5'
    render_mode = 'rgb_array' if video else None
    env = gym.make(env_name, render_mode=render_mode)
    obs, _ = env.reset()

    max_steps = int(duration_s * 50)
    total_reward = 0

    # Video writer
    out = None
    if video:
        frame = env.render().copy()
        brain_panel = newborn.draw_status_panel()
        cw = frame.shape[1] + brain_panel.shape[1]
        ch = max(frame.shape[0], brain_panel.shape[0])
        output_file = f"newborn_{body_type}.mp4"
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'),
                              25, (cw, ch))
        print(f"  Recording to {output_file}")

    baseline = 40 if body_type == "humanoid" else 119
    print(f"  Running {duration_s}s... (baseline: ~{baseline} steps)")

    for step in range(max_steps):
        action = newborn.step(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if video and step % 2 == 0:
            frame = env.render().copy()
            sensory = newborn.extract_sensory(obs)
            panel = newborn.draw_status_panel(sensory=sensory)

            cv2.putText(frame, f"Step: {step} ({step/50:.1f}s)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Reward: {total_reward:.0f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

            combined = np.zeros((ch, cw, 3), dtype=np.uint8)
            combined[:frame.shape[0], :frame.shape[1]] = frame
            combined[:panel.shape[0], frame.shape[1]:] = panel
            out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        if step % 250 == 0 and step > 0:
            print(f"    t={step/50:.0f}s  height={obs[0]:.2f}  tilt={obs[1]:.3f}  "
                  f"reward={total_reward:.0f}")

        if terminated or truncated:
            if video and out:
                fell = np.zeros((ch, cw, 3), dtype=np.uint8)
                fell[:] = (25, 25, 35)
                cv2.putText(fell, f"FELL at step {step} ({step/50:.1f}s)",
                            (cw // 6, ch // 2 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)
                cv2.putText(fell, f"Baseline: ~{baseline} steps ({baseline/50:.1f}s)",
                            (cw // 5, ch // 2 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
                cv2.putText(fell, f"Improvement: {(step+1)/baseline:.1f}x",
                            (cw // 5, ch // 2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)
                cv2.putText(fell, "Controller: Brainstem reflexes + Spinal CPG",
                            (cw // 6, ch // 2 + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 100), 1)
                cv2.putText(fell, "Cortex: observing (not controlling)",
                            (cw // 6, ch // 2 + 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 100, 255), 1)
                for _ in range(75):
                    out.write(cv2.cvtColor(fell, cv2.COLOR_RGB2BGR))
            break

    if out:
        out.release()
    env.close()

    print(f"\n  {'='*50}")
    print(f"  Survived: {step+1} steps ({(step+1)/50:.1f}s)")
    print(f"  Baseline: ~{baseline} steps ({baseline/50:.1f}s)")
    print(f"  Improvement: {(step+1)/baseline:.1f}x")
    print(f"  Reward: {total_reward:.0f}")
    print(f"  Controller: Brainstem + Spinal CPG (subcortical)")
    print(f"  Cortex: observer (not controlling)")

    if video:
        import shutil
        dst = f"/mnt/c/Users/lj880/Downloads/newborn_{body_type}.mp4"
        shutil.copy(f"newborn_{body_type}.mp4", dst)
        print(f"\n  Video copied to Windows Downloads: newborn_{body_type}.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanoid", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--spiking-cpg", action="store_true",
                        help="Use 80-neuron spiking CPG instead of Matsuoka rate model")
    parser.add_argument("--duration", type=int, default=30)
    args = parser.parse_args()

    body = "humanoid" if args.humanoid else "walker2d"
    run_demo(body_type=body, duration_s=args.duration, video=args.video,
             spiking_cpg=args.spiking_cpg)
