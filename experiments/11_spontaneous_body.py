"""v6: G=0.15, all 4 joints from motor/premotor regions (no BG saturation)."""
import time, numpy as np, torch
from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU
from encephagen.body.simple_body import SimpleBody
from encephagen.analysis.functional_roles import _classify_tvb76_regions

connectome = Connectome.from_bundled("tvb96")
groups = _classify_tvb76_regions(connectome.labels)
npr = 200; n_total = connectome.num_regions * npr

brain = SpikingBrainGPU(connectome=connectome, neurons_per_region=npr,
    global_coupling=0.15, ext_rate_factor=3.5, device="cuda")
body = SimpleBody()

motor_idx = groups.get("motor", [])
other_idx = groups.get("other", [])
sensory_idx = groups.get("sensory", [])

# Use motor + nearby cortical regions (NOT BG — too high firing)
joint_regions = motor_idx[:2] + other_idx[:2]
print(f"Joints: {[connectome.labels[i] for i in joint_regions]}")

state = brain.init_state(batch_size=1)
warmup_counts = torch.zeros(n_total, device="cuda")
with torch.no_grad():
    for _ in range(5000):
        state, spikes = brain.step(state)
        warmup_counts += spikes[0]
baselines = [warmup_counts[ri*npr:(ri+1)*npr].mean().item()/5000 for ri in joint_regions]
for i, ri in enumerate(joint_regions):
    print(f"  {connectome.labels[ri]}: baseline={baselines[i]:.4f} ({baselines[i]*10000:.0f} Hz)")

body_state = body.reset()
heights, torques_log, angles_log = [], [], []
t0 = time.time()

for step in range(2000):
    body_obs = body.get_sensory_input()
    external = torch.zeros(1, n_total, device="cuda")
    
    tilt = float(np.clip(body_obs[10], -1, 1))
    height = float(body_obs[8])
    
    # Joint angles + velocities
    for i, ri in enumerate(sensory_idx[:8]):
        if i < len(body_obs):
            s = ri * npr
            external[0, s:s+npr] = (float(body_obs[i]) + 1) * 8.0
    
    # Tilt → asymmetric drive (strongest signal)
    for ri in sensory_idx[8:12]:
        s = ri * npr
        if tilt > 0:
            external[0, s:s+npr//2] = tilt * 40.0
        else:
            external[0, s+npr//2:s+npr] = abs(tilt) * 40.0
    
    # Height danger
    danger = max(0, 0.8 - height)
    for ri in sensory_idx[:2]:
        external[0, ri*npr:(ri+1)*npr] += danger * 30.0
    
    # Fast feedback: 100 brain steps = 10ms
    joint_counts = [torch.zeros(npr, device="cuda") for _ in range(4)]
    with torch.no_grad():
        for _ in range(100):
            state, spikes = brain.step(state, external)
            for j, ri in enumerate(joint_regions):
                joint_counts[j] += spikes[0, ri*npr:(ri+1)*npr]
    
    torques = np.zeros(4)
    for j in range(4):
        rate = joint_counts[j].mean().item() / 100
        torques[j] = np.clip((rate - baselines[j]) * 400, -1, 1)
    
    body_state = body.step_n(torques, n=2)
    heights.append(body_state.torso_height)
    torques_log.append(torques.copy())
    angles_log.append(body_state.torso_angle)
    
    if body_state.is_fallen:
        print(f"  FELL at {step*0.01:.2f}s")
        break
    
    if (step+1) % 200 == 0:
        print(f"  {step*0.01:.1f}s  h={body_state.torso_height:.3f}  tilt={body_state.torso_angle:+.2f}")

elapsed = time.time() - t0
sim_time = len(heights) * 0.01
ta = np.array(torques_log); ha = np.array(heights); ang = np.array(angles_log)

print(f"\nSurvived {sim_time:.2f}s ({elapsed:.0f}s)")
print(f"Height: mean={ha.mean():.3f} std={ha.std():.3f} min={ha.min():.3f}")

names = ["r_hip","r_knee","l_hip","l_knee"]
for j in range(4):
    print(f"  {names[j]:<8} mean={ta[:,j].mean():+.3f} std={ta[:,j].std():.3f} range=[{ta[:,j].min():+.2f},{ta[:,j].max():+.2f}]")

if len(ang) > 10 and len(ta) > 10:
    print(f"\nCorrectiveness:")
    for j in range(4):
        c = np.corrcoef(ang, ta[:,j])[0,1]
        if not np.isnan(c):
            print(f"  {names[j]:<8} r={c:+.3f} {'CORRECTIVE' if c<-0.2 else 'destabilizing' if c>0.2 else ''}")

if len(ta) > 50:
    from scipy import signal
    print(f"\nRhythms:")
    for j in range(4):
        if ta[:,j].std() > 0.01:
            f, p = signal.welch(ta[:,j], fs=100, nperseg=min(128,len(ta)))
            pf = f[np.argmax(p)]; pp = p.max()
            if pp > 0.0001:
                print(f"  {names[j]:<8} {pf:.1f}Hz power={pp:.4f}")
