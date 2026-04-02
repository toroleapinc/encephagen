"""Tests for Milestone 4: Motor output and environment."""

import numpy as np

from encephagen.motor.decoder import MotorDecoder, MotorParams
from encephagen.environment.grid_world import GridWorld, GridWorldParams


# --- Motor decoder tests ---

def test_motor_creates():
    dec = MotorDecoder(n_neurons=200, seed=42)
    assert len(dec.action_groups) == 4
    total_neurons = sum(len(g) for g in dec.action_groups)
    assert total_neurons == 200


def test_motor_decode_with_spikes():
    dec = MotorDecoder(n_neurons=200, seed=42)
    # Make action group 0 (up) fire heavily
    spikes = np.zeros(200, dtype=bool)
    spikes[:50] = True  # First 50 neurons = action 0

    for _ in range(10):
        dec.update(spikes, dt=0.1)

    action = dec.decode_action()
    assert action == 0, f"Expected action 0 (up), got {action}"


def test_motor_decode_different_actions():
    """Different spike patterns should produce different actions."""
    dec = MotorDecoder(n_neurons=200, params=MotorParams(noise_sigma=0.0), seed=42)

    for target_action in range(4):
        dec.reset()
        spikes = np.zeros(200, dtype=bool)
        group = dec.action_groups[target_action]
        spikes[group] = True

        for _ in range(20):
            dec.update(spikes, dt=0.1)

        action = dec.decode_action()
        assert action == target_action, (
            f"Target action {target_action}, got {action}"
        )


def test_motor_get_rates():
    dec = MotorDecoder(n_neurons=200, seed=42)
    spikes = np.zeros(200, dtype=bool)
    spikes[100:150] = True  # Action group 2

    for _ in range(10):
        dec.update(spikes, dt=0.1)

    rates = dec.get_action_rates()
    assert rates.shape == (4,)
    assert rates[2] > rates[0]  # Group 2 should have highest rate


def test_motor_reset():
    dec = MotorDecoder(n_neurons=200, seed=42)
    dec.update(np.ones(200, dtype=bool), dt=0.1)
    dec.reset()
    rates = dec.get_action_rates()
    assert np.allclose(rates, 0)


# --- GridWorld tests ---

def test_gridworld_creates():
    env = GridWorld(seed=42)
    assert env.agent_pos.shape == (2,)
    assert env.target_pos.shape == (2,)
    assert not env.done


def test_gridworld_observe_shape():
    env = GridWorld(seed=42)
    obs = env.observe()
    assert obs.shape == (4,)
    assert np.all(obs >= 0)
    assert np.all(obs <= 1)


def test_gridworld_step():
    env = GridWorld(seed=42)
    obs, reward, done = env.step(0)  # Move up
    assert obs.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_gridworld_movement():
    """Actions should move the agent."""
    env = GridWorld(seed=42)
    pos_before = env.agent_pos.copy()
    env.step(0)  # Up
    pos_after = env.agent_pos.copy()
    assert not np.allclose(pos_before, pos_after), "Agent didn't move"


def test_gridworld_reaching_target():
    """Moving toward target should eventually reach it."""
    env = GridWorld(params=GridWorldParams(size=5.0, step_size=1.0, target_radius=1.0), seed=42)

    for _ in range(100):
        obs = env.observe()
        # Simple policy: move in direction of highest observation
        action = int(np.argmax(obs))
        _, reward, done = env.step(action)
        if done:
            break

    assert env.target_reached, (
        f"Failed to reach target. Distance: {env.distance_to_target:.2f}"
    )


def test_gridworld_reward_positive_when_closer():
    """Moving toward target should give positive reward."""
    env = GridWorld(seed=42)
    obs = env.observe()
    action = int(np.argmax(obs))  # Move toward target
    _, reward, _ = env.step(action)
    assert reward > 0, f"Moving toward target gave negative reward: {reward}"


def test_gridworld_reset():
    env = GridWorld(seed=42)
    pos1 = env.agent_pos.copy()
    env.step(0)
    env.reset()
    # After reset, position should be different from after step
    assert not env.done
    assert env.steps == 0


def test_gridworld_max_steps():
    """Episode should end after max_steps."""
    env = GridWorld(params=GridWorldParams(max_steps=5), seed=42)
    for i in range(10):
        _, _, done = env.step(0)
        if done:
            break
    assert done
    assert env.steps <= 5
