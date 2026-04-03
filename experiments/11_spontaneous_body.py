"""Spontaneous body v3: use multiple motor regions + stronger feedback loop."""
import time, numpy as np, torch
from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU
from encephagen.body.simple_body import SimpleBody
from encephagen.analysis.functional_roles import _classify_tvb76_regions

connectome = Connectome.from_bundled("tvb96")
groups = _classify_tvb76_regions(connectome.labels)

# Use 4 different regions for 4 joints (not just motor — spread across brain)
# This creates a distributed motor map
motor_regions = groups.get("motor", [])  # 2 regions (M1_R, M1_L)
other_regions = groups.get("other", [])
# Use motor + first 2 "other" regions for 4 joints
joint_regions = motor_regions[:2] + other_regions[:2]
print(f"Joint control regions: {[connectome.labels[i] for i in joint_regions]}")

sensory_idx = groups.get("sensory", [])
npr = 200
n_total = connectome.num_regions * npr

brain = SpikingBrainGPU(connectome=connectome, neurons_per_region=npr,
    global_coupling=0.15, ext_rate_factor=3.5, device="cuda")

body = SimpleBody()

# Warm up
print("Warming up (1s)...")
state = brain.init_state(batch_size=1)
joint_baseline_rates = []
with torch.no_grad():
    warmup_counts = torch.zeros(n_total, device="cuda")
    for _ in range(10000):
        state, spikes = brain.step(state)
        warmup_counts += spikes[0]
    for ri in joint_regions:
        s = ri * npr
        rate = warmup_counts[s:s+npr].mean().item() / 10000
        joint_baseline_rates.append(rate)
        print(f"  {connectome.labels[ri]}: baseline rate={rate:.4f}")

# Run with body
print(f"\nRunning body (10s)...")
body_state = body.reset()
heights, torques_log = [], []
t0 = time.time()

for step in range(200):
    # Sensory: stronger encoding, use joint angles directly
    body_obs = body.get_sensory_input()
    external = torch.zeros(1, n_total, device="cuda")

    # Map each joint angle to a different sensory region
    for i, ri in enumerate(sensory_idx[:8]):
        if i < len(body_obs):
            val = float(np.clip(body_obs[i], -1, 1))
            s = ri * npr
            # Asymmetric encoding: positive values excite first half,
            # negative values excite second half
            if val > 0:
                external[0, s:s+npr//2] = val * 10.0
            else:
                external[0, s+npr//2:s+npr] = abs(val) * 10.0

    # Brain steps
    joint_spike_counts = [torch.zeros(npr, device="cuda") for _ in range(4)]
    with torch.no_grad():
        for _ in range(500):  # 50ms
            state, spikes = brain.step(state, external)
            for j, ri in enumerate(joint_regions):
                s = ri * npr
                joint_spike_counts[j] += spikes[0, s:s+npr]

    # Motor output: deviation from baseline per joint region
    torques = np.zeros(4)
    for j in range(4):
        rate = joint_spike_counts[j].mean().item() / 500
        deviation = rate - joint_baseline_rates[j]
        # Different scaling for hip vs knee
        if j % 2 == 0:  # hip
            torques[j] = np.clip(deviation * 200, -1, 1)
        else:  # knee
            torques[j] = np.clip(deviation * 150, -1, 1)

    body_state = body.step_n(torques, n=10)
    heights.append(body_state.torso_height)
    torques_log.append(torques.copy())

    if body_state.is_fallen:
        print(f"  Fell at {step * 0.05:.1f}s")
        break

    if (step+1) % 20 == 0:
        print(f"  {step*0.05:.1f}s  h={body_state.torso_height:.3f}  "
              f"T=[{torques[0]:+.2f},{torques[1]:+.2f},{torques[2]:+.2f},{torques[3]:+.2f}]  "
              f"angle={body_state.torso_angle:+.2f}")

elapsed = time.time() - t0
sim_time = len(heights) * 0.05
ta = np.array(torques_log)

print(f"\n  Survived {sim_time:.1f}s ({elapsed:.0f}s wall)")
print(f"  Torque stats:")
names = ["r_hip", "r_knee", "l_hip", "l_knee"]
for j in range(4):
    print(f"    {names[j]:<8} mean={ta[:,j].mean():+.3f} std={ta[:,j].std():.3f} range=[{ta[:,j].min():+.2f},{ta[:,j].max():+.2f}]")

# Check if different joints do different things
print(f"\n  Joint differentiation:")
for j1 in range(4):
    for j2 in range(j1+1, 4):
        corr = np.corrcoef(ta[:,j1], ta[:,j2])[0,1] if len(ta) > 5 else 0
        print(f"    {names[j1]}-{names[j2]} correlation: {corr:+.3f}")

total_move = np.sum(np.abs(np.diff(np.array([body.get_state().joint_angles]))))
print(f"\n  Body moved: {'YES' if np.array(heights).std() > 0.05 else 'barely'}")
