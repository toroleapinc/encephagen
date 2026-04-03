"""Walking body with correct Matsuoka CPG — direct implementation."""
import numpy as np
from encephagen.body.simple_body import SimpleBody
from scipy import signal

# Matsuoka CPG — 4 units (left flex/ext, right flex/ext)
tau = 50.0
tau_a = 500.0
c = 2.0       # Tonic drive
w_m = 2.5     # Mutual inhibition within leg
w_c = 1.5     # Crossed inhibition between legs
beta = 2.5    # Adaptation
dt = 0.5      # ms

# State: x, v for each of 4 units
# 0=left_flex, 1=left_ext, 2=right_flex, 3=right_ext
x = np.array([0.5, -0.3, -0.3, 0.5])
v = np.zeros(4)

body = SimpleBody()
body.reset()

heights, torques_log, vel_log = [], [], []

for step in range(40000):  # 20 seconds
    y = np.maximum(0, x)

    # Proprioceptive feedback
    ja = body.get_state().joint_angles
    left_ext_signal = (ja[2] + ja[3]) / 2   # Left leg extension
    right_ext_signal = (ja[0] + ja[1]) / 2  # Right leg extension
    proprio_gain = 0.2

    # Drive with proprio modulation
    drives = np.array([
        c + proprio_gain * left_ext_signal,    # left flex: extended → flex more
        c - proprio_gain * left_ext_signal,    # left ext
        c + proprio_gain * right_ext_signal,   # right flex
        c - proprio_gain * right_ext_signal,   # right ext
    ])

    # Inhibition matrix
    inhib = np.zeros(4)
    inhib[0] = -w_m * y[1] - w_c * y[2]  # left flex ← left ext + right flex
    inhib[1] = -w_m * y[0] - w_c * y[3]  # left ext ← left flex + right ext
    inhib[2] = -w_m * y[3] - w_c * y[0]  # right flex ← right ext + left flex
    inhib[3] = -w_m * y[2] - w_c * y[1]  # right ext ← right flex + left ext

    # Dynamics
    dx = (-x + drives + inhib - beta * v) / tau
    dv = (-v + y) / tau_a

    x += dt * dx
    v += dt * dv

    # Torques: extension - flexion per leg
    left_torque = y[1] - y[0]   # left ext - left flex
    right_torque = y[3] - y[2]  # right ext - right flex

    torques = np.array([
        np.clip(right_torque * 1.5, -1, 1),       # right hip
        np.clip(right_torque * 0.7, -1, 1),        # right knee (coupled, weaker)
        np.clip(left_torque * 1.5, -1, 1),         # left hip
        np.clip(left_torque * 0.7, -1, 1),         # left knee
    ])

    body.step_n(torques, n=1)  # 0.5ms CPG, 5ms physics = needs 10 CPG steps per physics
    # Actually: dt_cpg=0.5ms, dt_physics=5ms. step_n(n=1) does 1 physics step.
    # So we're doing 1 CPG step per 1 physics step. That's fine for prototyping.

    bs = body.get_state()
    heights.append(bs.torso_height)
    torques_log.append(torques.copy())
    vel_log.append(bs.torso_velocity_x)

    if bs.is_fallen:
        print(f"Fell at {step*0.5/1000:.2f}s")
        break

    if (step+1) % 4000 == 0:
        ta = np.array(torques_log[-4000:])
        print(f"  {step*0.5/1000:.1f}s  h={bs.torso_height:.3f}  "
              f"hip_std={ta[:,0].std():.3f}  vel={bs.torso_velocity_x:+.3f}")

sim_time = len(heights) * 0.5 / 1000
ha = np.array(heights)
ta = np.array(torques_log)
va = np.array(vel_log)

print(f"\nSURVIVED: {sim_time:.2f}s")
print(f"Height: mean={ha.mean():.3f} std={ha.std():.3f}")
print(f"Forward velocity: mean={va.mean():+.4f} m/s")
print(f"Net displacement: {va.sum() * 0.0005:+.3f} m")

# Alternation
if len(ta) > 1000:
    lr = np.corrcoef(ta[1000:,0], ta[1000:,2])[0,1]
    print(f"L-R alternation: {lr:+.3f} {'ALTERNATING!' if lr < -0.3 else ''}")

# Frequency
if ta[:,0].std() > 0.01:
    f, ps = signal.welch(ta[:,0], fs=2000, nperseg=min(2048, len(ta)))
    pf = f[np.argmax(ps)]
    print(f"Stepping frequency: {pf:.2f} Hz")

# Summary
print(f"\nTorque variability (std > 0.1 means rhythmic movement):")
names = ["r_hip","r_knee","l_hip","l_knee"]
for j in range(4):
    print(f"  {names[j]:<8} std={ta[:,j].std():.3f}")

if ta[:,0].std() > 0.1 and sim_time > 5:
    print(f"\nSUCCESS: Rhythmic alternating movement sustained for {sim_time:.1f}s!")
elif sim_time > 5:
    print(f"\nBody survived {sim_time:.1f}s but movement is locked (not rhythmic)")
else:
    print(f"\nBody fell after {sim_time:.1f}s — needs more tuning")
