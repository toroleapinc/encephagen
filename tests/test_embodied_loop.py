"""Tests for embodied closed loop: brain controls MuJoCo body."""

import numpy as np
import pytest

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

from encephagen.connectome import Connectome

pytestmark = pytest.mark.skipif(not HAS_MUJOCO, reason="MuJoCo not installed")


def test_embodied_loop_creates():
    from encephagen.loop.embodied_loop import EmbodiedLoopRunner
    c = Connectome.from_bundled("toy20")
    runner = EmbodiedLoopRunner(
        c, neurons_per_region=100, enable_learning=False, seed=42,
    )
    assert runner.brain is not None
    assert runner.body is not None


def test_embodied_loop_runs_episode():
    from encephagen.loop.embodied_loop import EmbodiedLoopRunner
    c = Connectome.from_bundled("toy20")
    runner = EmbodiedLoopRunner(
        c, neurons_per_region=100, enable_learning=False,
        brain_steps_per_action=100, seed=42,
    )
    log = runner.run_episode(max_actions=5)
    assert log.steps > 0
    assert len(log.heights) == log.steps
    assert len(log.rewards) == log.steps


def test_embodied_loop_with_learning():
    from encephagen.loop.embodied_loop import EmbodiedLoopRunner
    c = Connectome.from_bundled("toy20")
    runner = EmbodiedLoopRunner(
        c, neurons_per_region=100, enable_learning=True,
        brain_steps_per_action=100, stdp_every=50, seed=42,
    )
    log = runner.run_episode(max_actions=3)
    assert log.steps > 0


def test_embodied_loop_body_responds():
    """Body height should change over an episode (not frozen)."""
    from encephagen.loop.embodied_loop import EmbodiedLoopRunner
    c = Connectome.from_bundled("toy20")
    runner = EmbodiedLoopRunner(
        c, neurons_per_region=100, enable_learning=False,
        brain_steps_per_action=100, seed=42,
    )
    log = runner.run_episode(max_actions=10)
    if len(log.heights) > 1:
        height_change = max(log.heights) - min(log.heights)
        assert height_change > 0.001, "Body didn't move at all"
