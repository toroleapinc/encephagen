"""Full Newborn — Maximum innate behavior coverage.

All neonatal reflexes on a Humanoid body. Zero learning. Pure 先天.

Behaviors:
  ✅ Righting reflex (tilt → correct)
  ✅ Moro reflex (drop → arms extend then flex)
  ✅ Startle reflex (sudden change → jerk)
  ✅ Palmar grasp (constant elbow flexion)
  ✅ ATNR/fencing (head turn → asymmetric arms)
  ✅ Stepping reflex (spiking CPG, 80 neurons)
  ✅ Breathing rhythm (abdomen CPG, ~40/min)
  ✅ General movements (multi-frequency fidgeting)
  ✅ Knee stabilization (constant slight flexion)
  ✅ Cortex observing (16K neurons, not controlling)

Usage:
  python newborn_full.py --video
"""

import argparse, json, time
import cv2
import numpy as np
import torch
import gymnasium as gym

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU
from encephagen.spinal.spiking_cpg import SpikingCPG
from encephagen.subcortical.full_brainstem import FullBrainstem


class FullNewborn:
    def __init__(self, device="cuda"):
        print("=" * 60)
        print("  FULL NEWBORN — All innate behaviors, zero learning")
        print("  16,080 spiking neurons | Humanoid body | Pure 先天")
        print("=" * 60)
        self.device = device

        # Brainstem (all reflexes)
        self.brainstem = FullBrainstem()

        # Spiking CPG
        print("  Building spiking CPG (80 neurons)...", flush=True)
        self.cpg = SpikingCPG(device=device)
        import os
        if os.path.exists("results/best_cpg_params_cmaes.npy"):
            self._load_cpg_params(np.load("results/best_cpg_params_cmaes.npy"))
        self.cpg_state = self.cpg.init_state()
        with torch.no_grad():
            for _ in range(3000):
                self.cpg_state, _, _ = self.cpg.step(self.cpg_state)

        # Cortex
        print("  Building cortex (16,000 neurons)...", flush=True)
        sc = np.load('src/encephagen/connectome/bundled/neurolib80_weights.npy')
        tl = np.load('src/encephagen/connectome/bundled/neurolib80_tract_lengths.npy')
        labels = json.load(open('src/encephagen/connectome/bundled/neurolib80_labels.json'))
        tau_labels = json.load(open('src/encephagen/connectome/bundled/neurolib80_t1t2_labels.json'))
        self.tau_m = np.load('src/encephagen/connectome/bundled/neurolib80_tau_m.npy')
        c = Connectome(sc, labels); c.tract_lengths = tl

        self.cortex = SpikingBrainGPU(connectome=c, neurons_per_region=200,
            internal_conn_prob=0.05, between_conn_prob=0.03,
            global_coupling=5.0, ext_rate_factor=3.5,
            pfc_regions=[], device=device,
            use_delays=True, conduction_velocity=3.5, use_t1t2_gradient=True)
        self.cortex_state = self.cortex.init_state(batch_size=1)
        with torch.no_grad():
            for _ in range(2000):
                self.cortex_state, _ = self.cortex.step(self.cortex_state)

        self.cortex_groups = {}
        for key, pats in [('Somatosens', ['Postcentral']), ('Motor', ['Precentral']),
                           ('Visual', ['Calcarine']), ('Frontal', ['Frontal_Sup']),
                           ('Temporal', ['Temporal_Mid']), ('Parietal', ['Parietal_Sup'])]:
            self.cortex_groups[key] = [i for i, l in enumerate(tau_labels) if any(p in l for p in pats)]
        self.soma_starts = [ri * 200 for ri in self.cortex_groups['Somatosens']]
        self.cortex_rates = {}

        # CPG params for custom step
        self._cpg_beta = 2.0; self._cpg_tau = 100.0; self._cpg_reset_v = 4.0

        print("  Ready.\n")

    def _load_cpg_params(self, params):
        """Apply CMA-ES params to CPG."""
        (w_mfe, w_mef, w_rm, w_vi, w_vo, w_pd, w_pi,
         d_f, d_e, d_m, d_v, beta, tau, rv) = params
        cpg = self.cpg; W = cpg.W.clone()
        for side in ['L','R']:
            o = 'R' if side=='L' else 'L'
            fr=cpg.idx[f'{side}_flex_rg']; er=cpg.idx[f'{side}_ext_rg']
            fm=cpg.idx[f'{side}_flex_mn']; em=cpg.idx[f'{side}_ext_mn']
            v1=cpg.idx[f'{side}_v1']; v2b=cpg.idx[f'{side}_v2b']
            v0d=cpg.idx[f'{side}_v0d']; cfr=cpg.idx[f'{o}_flex_rg']
            def sw(s,d,w,n=True):
                ns=s.stop-s.start; wn=w/max(ns,1) if n else w
                for i in range(s.start,s.stop):
                    for j in range(d.start,d.stop):
                        if i!=j: W[i,j]=wn
            sw(fr,er,w_mfe); sw(er,fr,w_mef)
            sw(fr,fm,w_rm,False); sw(er,em,w_rm,False)
            sw(fr,v0d,w_vi,False); sw(v0d,cfr,w_vo,False)
            sw(fr,v1,w_pd,False); sw(er,v2b,w_pd,False)
            sw(v1,em,w_pi,False); sw(v2b,fm,w_pi,False)
        cpg.W = W
        cpg.tonic_drive = torch.zeros(cpg.n_total, device=self.device)
        for s in ['L','R']:
            cpg.tonic_drive[cpg.idx[f'{s}_flex_rg']]=d_f
            cpg.tonic_drive[cpg.idx[f'{s}_ext_rg']]=d_e
            cpg.tonic_drive[cpg.idx[f'{s}_flex_mn']]=d_m
            cpg.tonic_drive[cpg.idx[f'{s}_ext_mn']]=d_m
            cpg.tonic_drive[cpg.idx[f'{s}_v1']]=5.0
            cpg.tonic_drive[cpg.idx[f'{s}_v2b']]=5.0
            cpg.tonic_drive[cpg.idx[f'{s}_v0d']]=d_v
        self._cpg_beta=beta; self._cpg_tau=tau; self._cpg_reset_v=rv

    def step(self, obs):
        """Full newborn step with all reflexes."""
        # Brainstem reflexes → 17-dim action
        action = self.brainstem.process(obs)

        # Spiking CPG → leg rhythm
        drive = self.brainstem.stepping_drive
        l_acc = r_acc = 0.0
        cpg = self.cpg
        with torch.no_grad():
            for _ in range(20):
                st = self.cpg_state
                v=st['v']; rf=st['refrac']; isyn=st['i_syn']; ad=st['adaptation']
                dr = cpg.tonic_drive * (1.0 + drive * 0.5)
                noise = torch.randn(cpg.n_total, device=self.device) * 0.5
                it = isyn + dr - ad + noise
                act = rf <= 0; dv = (-v + it) / cpg.tau_m
                v = v + cpg.dt * dv * act.float()
                sp = (v >= cpg.v_threshold) & act
                rv = torch.zeros_like(v)
                for s in ['L','R']:
                    rv[cpg.idx[f'{s}_flex_rg']] = self._cpg_reset_v
                    rv[cpg.idx[f'{s}_ext_rg']] = self._cpg_reset_v
                v = torch.where(sp, rv, v)
                rf = torch.where(sp, torch.full_like(rf, 1.0), rf)
                rf = torch.clamp(rf - cpg.dt, min=0)
                ad = ad * np.exp(-cpg.dt / self._cpg_tau) + sp.float() * self._cpg_beta
                si = cpg.W @ sp.float()
                isyn = isyn * np.exp(-cpg.dt / 5.0) + si
                self.cpg_state = {'v':v,'refrac':rf,'i_syn':isyn,'adaptation':ad}
                lf = sp[cpg.idx['L_flex_mn']].float().mean().item()
                le = sp[cpg.idx['L_ext_mn']].float().mean().item()
                rf2 = sp[cpg.idx['R_flex_mn']].float().mean().item()
                re = sp[cpg.idx['R_ext_mn']].float().mean().item()
                l_acc += (le - lf); r_acc += (re - rf2)

        # Add CPG to legs
        cpg_scale = 0.15 * drive
        action[5] += np.clip(r_acc / 20 * cpg_scale, -0.15, 0.15)  # right hip
        action[6] += np.clip(r_acc / 20 * cpg_scale * 0.7, -0.1, 0.1)  # right knee
        action[9] += np.clip(l_acc / 20 * cpg_scale, -0.15, 0.15)  # left hip
        action[10] += np.clip(l_acc / 20 * cpg_scale * 0.7, -0.1, 0.1)  # left knee

        # Cortex observes
        ext = torch.zeros(1, 16000, device=self.device)
        signal = abs(obs[1]) * 50 + max(0, (1.3-obs[0])) * 40
        for s in self.soma_starts:
            ext[0, s:s+200] = signal
        with torch.no_grad():
            for _ in range(10):
                self.cortex_state, sp = self.cortex.step(self.cortex_state, ext)
        for gn, gi in self.cortex_groups.items():
            t = sum(sp[0,ri*200:(ri+1)*200].sum().item() for ri in gi)
            self.cortex_rates[gn] = t / (len(gi) * 200)

        return np.clip(action, -0.4, 0.4)

    def draw_panel(self, width=420, height=480):
        """Status panel showing all active systems."""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (25, 25, 35)

        cv2.putText(panel, "NEWBORN BRAIN (16,080 neurons)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        cv2.putText(panel, "All innate behaviors | Zero learning", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)

        y = 65
        # Brainstem reflexes
        cv2.putText(panel, "BRAINSTEM REFLEXES", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        y += 20
        reflexes = self.brainstem.get_active_reflexes()
        for name, intensity in reflexes.items():
            bar_w = int(min(abs(intensity) * 150, 150))
            color = (100, 255, 150) if intensity > 0.1 else (50, 50, 60)
            cv2.putText(panel, name, (15, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)
            cv2.rectangle(panel, (130, y), (130 + max(bar_w, 1), y + 11), color, -1)
            y += 16

        y += 8; cv2.line(panel, (10, y), (width-10, y), (50, 50, 60), 1); y += 12

        # Spiking CPG
        cv2.putText(panel, "SPINAL CPG (80 spiking neurons)", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 200, 255), 1)
        y += 18
        for side, label in [('L','left'), ('R','right')]:
            for mn in ['flex','ext']:
                idx = self.cpg.idx[f'{side}_{mn}_mn']
                v = self.cpg_state['v'][idx].mean().item() / 8.0
                bar_w = int(min(abs(v) * 120, 120))
                color = (100, 200, 255) if v > 0.3 else (50, 50, 60)
                cv2.putText(panel, f"{label} {mn}", (15, y+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (140, 140, 140), 1)
                cv2.rectangle(panel, (110, y), (110+max(bar_w,1), y+10), color, -1)
                y += 14

        y += 8; cv2.line(panel, (10, y), (width-10, y), (50, 50, 60), 1); y += 12

        # Cortex
        cv2.putText(panel, "CORTEX (16K neurons, observing)", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 100, 255), 1)
        y += 18
        for gn in ['Somatosens', 'Motor', 'Visual', 'Frontal', 'Parietal', 'Temporal']:
            r = self.cortex_rates.get(gn, 0)
            tau = np.mean([self.tau_m[i] for i in self.cortex_groups.get(gn, [0])])
            bar_w = int(min(r * 200 * 15, 120))
            cv2.putText(panel, gn, (15, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
            cv2.rectangle(panel, (110, y), (110+max(bar_w,1), y+10), (120, 80, 200), -1)
            cv2.putText(panel, f"{tau:.0f}ms", (240, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (80,80,80), 1)
            y += 14

        return panel


def run(duration_s=20, video=False):
    nb = FullNewborn()
    env = gym.make('Humanoid-v5', render_mode='rgb_array' if video else None)
    obs, _ = env.reset()
    max_steps = int(duration_s * 50)

    out = None
    if video:
        frame = env.render().copy()
        panel = nb.draw_panel()
        cw = frame.shape[1] + panel.shape[1]
        ch = max(frame.shape[0], panel.shape[0])
        out_file = "newborn_humanoid_full.mp4"
        out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), 25, (cw, ch))

    total_reward = 0
    print(f"  Running {duration_s}s... (baseline: ~40 steps)")

    for step in range(max_steps):
        action = nb.step(obs)
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward

        if video and step % 2 == 0:
            frame = env.render().copy()
            panel = nb.draw_panel()
            cv2.putText(frame, f"Step {step} ({step/50:.1f}s)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Reward: {total_reward:.0f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
            combined = np.zeros((ch, cw, 3), dtype=np.uint8)
            combined[:frame.shape[0], :frame.shape[1]] = frame
            combined[:panel.shape[0], frame.shape[1]:] = panel
            out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        if step % 250 == 0 and step > 0:
            reflexes = nb.brainstem.get_active_reflexes()
            active = [k for k, v in reflexes.items() if v > 0.1]
            print(f"    t={step/50:.0f}s h={obs[0]:.2f} tilt={obs[1]:.3f} "
                  f"active: {', '.join(active)}")

        if term or trunc:
            if out:
                fell = np.zeros((ch, cw, 3), dtype=np.uint8); fell[:] = (25, 25, 35)
                cv2.putText(fell, f"FELL at step {step} ({step/50:.1f}s)",
                            (cw//6, ch//2-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,100,255), 2)
                cv2.putText(fell, f"Baseline: ~40 steps (0.8s) | Improvement: {(step+1)/40:.1f}x",
                            (cw//7, ch//2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
                cv2.putText(fell, "Controller: Brainstem reflexes + 80-neuron spiking CPG",
                            (cw//7, ch//2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,200,100), 1)
                cv2.putText(fell, "Cortex: 16,000 neurons observing (not controlling)",
                            (cw//7, ch//2+65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,100,255), 1)
                cv2.putText(fell, "Learning: NONE (pure innate behavior)",
                            (cw//7, ch//2+90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,255,100), 1)
                for _ in range(75): out.write(cv2.cvtColor(fell, cv2.COLOR_RGB2BGR))
            break

    if out:
        out.release()
        import shutil
        shutil.copy(out_file, f"/mnt/c/Users/lj880/Downloads/{out_file}")
        print(f"\n  Video: {out_file} → Downloads")
    env.close()

    print(f"\n  {'='*50}")
    print(f"  Survived: {step+1} steps ({(step+1)/50:.1f}s)")
    print(f"  Baseline: ~40 steps (0.8s)")
    print(f"  Improvement: {(step+1)/40:.1f}x")
    print(f"  Total reward: {total_reward:.0f}")
    print(f"  Active reflexes: righting, Moro, startle, grasp, ATNR,")
    print(f"    breathing, general movements, stepping (spiking CPG)")
    print(f"  Cortex: 16,000 neurons observing, not controlling")
    print(f"  Learning: NONE — pure innate behavior (先天)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--duration", type=int, default=20)
    args = parser.parse_args()
    run(duration_s=args.duration, video=args.video)
