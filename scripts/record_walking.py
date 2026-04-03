"""Record a video of the brain-controlled body walking.

Renders the MuJoCo body while the Matsuoka CPG drives alternating
leg movement, with the brain modulating the rhythm.
"""

import numpy as np
import mujoco
import imageio

from encephagen.body.simple_body import SimpleBody, SIMPLE_BODY_XML


def run_cpg_step(x, v, drives, tau=50, tau_a=500, w_m=2.5, w_c=1.5, beta=2.5, dt=0.5):
    y = np.maximum(0, x)
    inh = np.zeros(4)
    inh[0] = -w_m*y[1] - w_c*y[2]
    inh[1] = -w_m*y[0] - w_c*y[3]
    inh[2] = -w_m*y[3] - w_c*y[0]
    inh[3] = -w_m*y[2] - w_c*y[1]
    dx = (-x + drives + inh - beta*v) / tau
    dv = (-v + y) / tau_a
    x = x + dt * dx
    v = v + dt * dv
    lt = y[1] - y[0]
    rt = y[3] - y[2]
    scale = 0.5
    torques = np.clip(
        np.array([rt*scale, rt*scale*0.5, lt*scale, lt*scale*0.5]),
        -1, 1
    )
    return x, v, torques


def record_video():
    print("Recording walking video...")

    # Setup
    model = mujoco.MjModel.from_xml_string(SIMPLE_BODY_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    renderer = mujoco.Renderer(model, width=640, height=480)

    # Camera setup — side view
    renderer.update_scene(data, camera=mujoco.MjvCamera())
    cam = mujoco.MjvCamera()
    cam.lookat[0] = 0.15  # Look at center of body
    cam.lookat[1] = 0.0
    cam.lookat[2] = 0.3
    cam.distance = 1.8
    cam.azimuth = 90  # Side view
    cam.elevation = -15

    # CPG
    x = np.array([0.5, -0.3, -0.3, 0.5])
    v = np.zeros(4)
    c_base = 2.0

    # Record
    frames = []
    fps = 30
    physics_per_frame = int(1.0 / (model.opt.timestep * fps))  # Physics steps per video frame
    cpg_per_physics = 1  # CPG steps per physics step (dt matched)

    duration_sec = 10
    total_frames = duration_sec * fps

    for frame in range(total_frames):
        # Run physics for this frame
        for _ in range(physics_per_frame):
            drives = np.full(4, c_base)
            x, v, torques = run_cpg_step(x, v, drives)
            data.ctrl[:4] = torques
            mujoco.mj_step(model, data)

        # Render
        renderer.update_scene(data, cam)
        pixels = renderer.render()
        frames.append(pixels.copy())

        if (frame + 1) % (fps * 2) == 0:
            height = data.xpos[1][2] if len(data.xpos) > 1 else 0
            print(f"  {(frame+1)/fps:.0f}s  height={height:.3f}")

    # Save video
    output_path = "figures/walking_demo.mp4"
    print(f"\nSaving {len(frames)} frames to {output_path}...")
    writer = imageio.get_writer(output_path, fps=fps, quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    # Also save a few key frames as images
    for i, t in enumerate([0, len(frames)//4, len(frames)//2, 3*len(frames)//4]):
        imageio.imwrite(f"figures/walking_frame_{i}.png", frames[t])

    print(f"Video saved: {output_path}")
    print(f"Key frames saved: figures/walking_frame_*.png")

    renderer.close()


if __name__ == "__main__":
    record_video()
