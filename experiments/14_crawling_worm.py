"""Crawling body — inherently stable, doesn't need balance.
Like C. elegans: a segmented body that produces locomotion through
alternating contraction waves."""
import numpy as np, mujoco, imageio

CRAWLER_XML = """
<mujoco model="crawler">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light diffuse=".8 .8 .8" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="10 10 0.1" rgba=".9 .9 .9 1" friction="1.5 0.5 0.1"/>

    <!-- Segmented worm/snake body on the ground -->
    <body name="seg0" pos="0 0 0.08">
      <freejoint/>
      <geom type="capsule" size="0.04" fromto="0 0 0 0.12 0 0" mass="0.2" rgba="0.4 0.6 0.8 1"/>

      <body name="seg1" pos="0.12 0 0">
        <joint name="j1" type="hinge" axis="0 0 1" range="-45 45" damping="0.05"
               stiffness="0.2" springref="0"/>
        <geom type="capsule" size="0.035" fromto="0 0 0 0.12 0 0" mass="0.15" rgba="0.5 0.7 0.4 1"/>

        <body name="seg2" pos="0.12 0 0">
          <joint name="j2" type="hinge" axis="0 0 1" range="-45 45" damping="0.05"
                 stiffness="0.2" springref="0"/>
          <geom type="capsule" size="0.035" fromto="0 0 0 0.12 0 0" mass="0.15" rgba="0.4 0.6 0.8 1"/>

          <body name="seg3" pos="0.12 0 0">
            <joint name="j3" type="hinge" axis="0 0 1" range="-45 45" damping="0.05"
                   stiffness="0.2" springref="0"/>
            <geom type="capsule" size="0.035" fromto="0 0 0 0.12 0 0" mass="0.15" rgba="0.5 0.7 0.4 1"/>

            <body name="seg4" pos="0.12 0 0">
              <joint name="j4" type="hinge" axis="0 0 1" range="-45 45" damping="0.05"
                     stiffness="0.2" springref="0"/>
              <geom type="capsule" size="0.03" fromto="0 0 0 0.1 0 0" mass="0.1" rgba="0.4 0.6 0.8 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" joint="j1" gear="3" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="m2" joint="j2" gear="3" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="m3" joint="j3" gear="3" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="m4" joint="j4" gear="3" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(CRAWLER_XML)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# Simple traveling wave CPG — each joint oscillates with a phase offset
# This produces the S-shaped undulation like a snake/worm
freq = 1.5  # Hz
phase_offsets = [0, np.pi/2, np.pi, 3*np.pi/2]  # Traveling wave
amplitude = 0.8

renderer = mujoco.Renderer(model, width=640, height=480)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.3, 0, 0.08]
cam.distance = 1.2
cam.azimuth = 0   # Top-down-ish
cam.elevation = -40

frames = []
fps = 30
steps_per_frame = int(1.0 / (0.005 * fps))
positions = []

for frame_i in range(fps * 10):
    t = frame_i / fps  # seconds

    for _ in range(steps_per_frame):
        sim_t = data.time
        for j in range(4):
            data.ctrl[j] = amplitude * np.sin(2 * np.pi * freq * sim_t + phase_offsets[j])
        mujoco.mj_step(model, data)

    # Track head position
    head_pos = data.xpos[1].copy() if len(data.xpos) > 1 else np.zeros(3)
    positions.append(head_pos.copy())

    renderer.update_scene(data, cam)
    frames.append(renderer.render().copy())

    if (frame_i+1) % (fps*2) == 0:
        print(f"  {t:.0f}s  pos=[{head_pos[0]:+.3f},{head_pos[1]:+.3f}]")

renderer.close()

# Save
out = "/home/lj_wsl/encephagen/figures/crawling.mp4"
writer = imageio.get_writer(out, fps=fps, quality=8)
for f in frames: writer.append_data(f)
writer.close()

for i in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4]:
    if i < len(frames):
        imageio.imwrite(f"/home/lj_wsl/encephagen/figures/crawl_{i}.png", frames[i])

pos = np.array(positions)
displacement = np.sqrt((pos[-1,0]-pos[0,0])**2 + (pos[-1,1]-pos[0,1])**2)
print(f"\nVideo: {out}")
print(f"Start: [{pos[0,0]:.3f}, {pos[0,1]:.3f}]")
print(f"End:   [{pos[-1,0]:.3f}, {pos[-1,1]:.3f}]")
print(f"Displacement: {displacement:.3f} m")
if displacement > 0.05:
    print(f"IT CRAWLS! Forward displacement: {displacement:.3f} m in 10s")
else:
    print(f"No significant forward movement")
