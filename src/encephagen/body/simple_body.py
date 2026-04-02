"""SimpleBody: a minimal MuJoCo body for the brain to control.

A torso with 2 legs, each with 2 joints (hip + knee).
The brain controls joint torques through motor regions and
receives proprioceptive feedback (joint angles, velocities)
through sensory regions.

The simplest possible embodiment that tests whether a
brain-structured controller can learn to produce movement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


# MuJoCo XML for a simple 2-legged body
SIMPLE_BODY_XML = """
<mujoco model="simple_walker">
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="10 10 0.1" rgba=".9 .9 .9 1"/>

    <!-- Torso -->
    <body name="torso" pos="0 0 0.6">
      <joint name="root_x" type="slide" axis="1 0 0"/>
      <joint name="root_z" type="slide" axis="0 0 1"/>
      <joint name="root_rot" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.15" rgba="0.3 0.5 0.8 1"
            fromto="0 0 0 0.3 0 0" mass="1.0"/>

      <!-- Right leg -->
      <body name="right_thigh" pos="0 0 0">
        <joint name="right_hip" type="hinge" axis="0 1 0"
               range="-60 60" damping="0.5"/>
        <geom type="capsule" size="0.04 0.15" rgba="0.8 0.3 0.3 1"
              fromto="0 0 0 0 0 -0.3" mass="0.5"/>

        <body name="right_shin" pos="0 0 -0.3">
          <joint name="right_knee" type="hinge" axis="0 1 0"
                 range="-90 0" damping="0.5"/>
          <geom type="capsule" size="0.03 0.15" rgba="0.8 0.5 0.3 1"
                fromto="0 0 0 0 0 -0.3" mass="0.3"/>
          <!-- Foot -->
          <geom type="sphere" size="0.05" pos="0 0 -0.3"
                rgba="0.5 0.5 0.5 1" mass="0.1"/>
        </body>
      </body>

      <!-- Left leg -->
      <body name="left_thigh" pos="0.3 0 0">
        <joint name="left_hip" type="hinge" axis="0 1 0"
               range="-60 60" damping="0.5"/>
        <geom type="capsule" size="0.04 0.15" rgba="0.3 0.8 0.3 1"
              fromto="0 0 0 0 0 -0.3" mass="0.5"/>

        <body name="left_shin" pos="0 0 -0.3">
          <joint name="left_knee" type="hinge" axis="0 1 0"
                 range="-90 0" damping="0.5"/>
          <geom type="capsule" size="0.03 0.15" rgba="0.3 0.8 0.5 1"
                fromto="0 0 0 0 0 -0.3" mass="0.3"/>
          <!-- Foot -->
          <geom type="sphere" size="0.05" pos="0 0 -0.3"
                rgba="0.5 0.5 0.5 1" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="right_hip_motor" joint="right_hip" gear="20"
           ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="right_knee_motor" joint="right_knee" gear="20"
           ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="left_hip_motor" joint="left_hip" gear="20"
           ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="left_knee_motor" joint="left_knee" gear="20"
           ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>

  <sensor>
    <jointpos name="right_hip_pos" joint="right_hip"/>
    <jointpos name="right_knee_pos" joint="right_knee"/>
    <jointpos name="left_hip_pos" joint="left_hip"/>
    <jointpos name="left_knee_pos" joint="left_knee"/>
    <jointvel name="right_hip_vel" joint="right_hip"/>
    <jointvel name="right_knee_vel" joint="right_knee"/>
    <jointvel name="left_hip_vel" joint="left_hip"/>
    <jointvel name="left_knee_vel" joint="left_knee"/>
    <framepos name="torso_pos" objtype="body" objname="torso"/>
    <framequat name="torso_quat" objtype="body" objname="torso"/>
  </sensor>
</mujoco>
"""


@dataclass
class BodyState:
    """Current state of the body."""

    joint_angles: np.ndarray      # [4] hip/knee angles (rad)
    joint_velocities: np.ndarray  # [4] hip/knee velocities (rad/s)
    torso_height: float           # Height of torso above ground
    torso_velocity_x: float       # Forward velocity
    torso_angle: float            # Torso tilt angle (rad)
    is_fallen: bool               # Whether torso is below threshold


class SimpleBody:
    """A minimal 2-legged MuJoCo body.

    4 joints: right_hip, right_knee, left_hip, left_knee
    4 actuators: torques for each joint (range [-1, 1])
    Sensors: joint angles, velocities, torso position/orientation
    """

    def __init__(self):
        if not HAS_MUJOCO:
            raise ImportError("MuJoCo is required for SimpleBody. Install with: conda install -c conda-forge mujoco")

        self.model = mujoco.MjModel.from_xml_string(SIMPLE_BODY_XML)
        self.data = mujoco.MjData(self.model)

        self.n_actuators = self.model.nu  # 4
        self.n_joints = 4  # controlled joints (not root)
        self._fall_threshold = 0.2  # torso height below this = fallen

        # Joint name to index mapping
        self._joint_names = ["right_hip", "right_knee", "left_hip", "left_knee"]
        self._actuator_names = [f"{j}_motor" for j in self._joint_names]

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> BodyState:
        """Reset body to standing position."""
        mujoco.mj_resetData(self.model, self.data)
        # Small random perturbation to break symmetry
        self.data.qpos[:] += np.random.uniform(-0.01, 0.01, size=self.data.qpos.shape)
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def step(self, torques: np.ndarray) -> BodyState:
        """Apply joint torques and advance physics one timestep.

        Args:
            torques: [4] array of control signals in [-1, 1].

        Returns:
            Updated body state.
        """
        torques = np.clip(torques, -1.0, 1.0)
        self.data.ctrl[:self.n_actuators] = torques
        mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def step_n(self, torques: np.ndarray, n: int = 10) -> BodyState:
        """Apply torques and advance physics n timesteps.

        Useful for matching brain simulation speed to physics speed.
        Physics dt=0.005s, so n=10 gives 50ms per brain action.
        """
        torques = np.clip(torques, -1.0, 1.0)
        self.data.ctrl[:self.n_actuators] = torques
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def get_state(self) -> BodyState:
        """Read current body state from sensors."""
        # Joint angles (4 controlled joints)
        angles = np.array(self.data.sensordata[:4], dtype=np.float64)
        velocities = np.array(self.data.sensordata[4:8], dtype=np.float64)

        # Torso position (x, y, z from framepos sensor)
        torso_pos = self.data.sensordata[8:11]
        torso_height = float(torso_pos[2])

        # Torso velocity (from qvel of root joints)
        torso_vel_x = float(self.data.qvel[0])  # root_x velocity

        # Torso angle (from root_rot joint)
        torso_angle = float(self.data.qpos[2])  # root_rot

        is_fallen = torso_height < self._fall_threshold

        return BodyState(
            joint_angles=angles,
            joint_velocities=velocities,
            torso_height=torso_height,
            torso_velocity_x=torso_vel_x,
            torso_angle=torso_angle,
            is_fallen=is_fallen,
        )

    def get_sensory_input(self) -> np.ndarray:
        """Get sensory observation for the brain.

        Returns:
            [12] array:
            - [0:4] joint angles (normalized to [-1, 1])
            - [4:8] joint velocities (clipped and normalized)
            - [8] torso height (normalized)
            - [9] torso forward velocity (normalized)
            - [10] torso angle (normalized)
            - [11] is_fallen (0 or 1)
        """
        state = self.get_state()
        obs = np.zeros(12, dtype=np.float64)

        # Joint angles: normalize by range
        obs[0:4] = state.joint_angles / np.pi  # Roughly [-1, 1]

        # Joint velocities: clip and normalize
        obs[4:8] = np.clip(state.joint_velocities / 10.0, -1, 1)

        # Torso
        obs[8] = np.clip(state.torso_height / 1.0, 0, 2)  # Height normalized
        obs[9] = np.clip(state.torso_velocity_x / 2.0, -1, 1)  # Velocity normalized
        obs[10] = np.clip(state.torso_angle / np.pi, -1, 1)  # Angle normalized
        obs[11] = 1.0 if state.is_fallen else 0.0

        return obs

    def compute_reward(self, state: BodyState) -> float:
        """Compute reward for the current state.

        Rewards:
        - Staying upright (torso height > threshold)
        - Moving forward (positive x velocity)
        - Penalize falling
        - Penalize excessive torso tilt
        """
        reward = 0.0

        # Upright bonus
        if not state.is_fallen:
            reward += 0.5

        # Height bonus (more reward for being tall)
        reward += np.clip(state.torso_height - 0.3, 0, 0.5)

        # Forward velocity bonus
        reward += np.clip(state.torso_velocity_x * 0.5, -0.5, 1.0)

        # Tilt penalty
        reward -= abs(state.torso_angle) * 0.3

        # Fall penalty
        if state.is_fallen:
            reward -= 5.0

        return float(reward)

    @property
    def physics_dt(self) -> float:
        """Physics timestep in seconds."""
        return float(self.model.opt.timestep)
