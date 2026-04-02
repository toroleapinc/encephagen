"""Tests for Milestone 6: Virtual body (MuJoCo)."""

import numpy as np
import pytest

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

from encephagen.body.simple_body import SimpleBody, BodyState


pytestmark = pytest.mark.skipif(not HAS_MUJOCO, reason="MuJoCo not installed")


def test_body_creates():
    body = SimpleBody()
    assert body.n_actuators == 4
    assert body.n_joints == 4


def test_body_reset():
    body = SimpleBody()
    state = body.reset()
    assert isinstance(state, BodyState)
    assert state.joint_angles.shape == (4,)
    assert state.joint_velocities.shape == (4,)
    assert state.torso_height > 0


def test_body_step():
    body = SimpleBody()
    body.reset()
    torques = np.zeros(4)
    state = body.step(torques)
    assert isinstance(state, BodyState)


def test_body_step_n():
    body = SimpleBody()
    body.reset()
    torques = np.zeros(4)
    state = body.step_n(torques, n=10)
    assert isinstance(state, BodyState)


def test_body_torques_affect_state():
    """Applying torques should change joint angles."""
    body = SimpleBody()
    state0 = body.reset()
    angles0 = state0.joint_angles.copy()

    # Apply strong torque to right hip
    torques = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(100):
        state = body.step(torques)

    assert not np.allclose(state.joint_angles, angles0, atol=0.01), (
        "Torques didn't change joint angles"
    )


def test_body_falls_without_control():
    """Without any control, body should eventually fall."""
    body = SimpleBody()
    body.reset()

    for _ in range(1000):
        state = body.step(np.zeros(4))

    # After 1000 steps with no control, height should decrease
    assert state.torso_height < 0.6, (
        f"Body didn't fall: height={state.torso_height:.2f}"
    )


def test_body_sensory_input_shape():
    body = SimpleBody()
    body.reset()
    obs = body.get_sensory_input()
    assert obs.shape == (12,)
    assert not np.isnan(obs).any()


def test_body_reward_upright():
    """Upright body should get positive reward."""
    body = SimpleBody()
    state = body.reset()
    reward = body.compute_reward(state)
    assert reward > 0, f"Upright body got negative reward: {reward}"


def test_body_reward_fallen():
    """Fallen body should get negative reward."""
    body = SimpleBody()
    body.reset()

    # Force fall
    for _ in range(2000):
        body.step(np.zeros(4))

    state = body.get_state()
    if state.is_fallen:
        reward = body.compute_reward(state)
        assert reward < 0, f"Fallen body got positive reward: {reward}"


def test_body_torque_clipping():
    """Torques outside [-1, 1] should be clipped."""
    body = SimpleBody()
    body.reset()
    # Should not crash with extreme values
    state = body.step(np.array([100.0, -100.0, 50.0, -50.0]))
    assert isinstance(state, BodyState)
