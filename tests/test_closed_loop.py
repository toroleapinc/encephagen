"""Tests for Milestone 5: Closed-loop brain-environment interaction."""

import numpy as np

from encephagen.connectome import Connectome
from encephagen.loop.closed_loop import ClosedLoopRunner


def test_closed_loop_creates():
    c = Connectome.from_bundled("toy20")
    runner = ClosedLoopRunner(c, neurons_per_region=100, enable_learning=False, seed=42)
    assert runner.brain is not None
    assert runner.env is not None


def test_closed_loop_runs_episode():
    """A single episode should complete without crashing."""
    c = Connectome.from_bundled("toy20")
    runner = ClosedLoopRunner(
        c, neurons_per_region=100, global_coupling=0.05,
        enable_learning=False, action_every_ms=20.0, seed=42,
    )
    log = runner.run_episode()
    assert log.steps > 0
    assert len(log.distances) == log.steps
    assert len(log.actions) == log.steps


def test_closed_loop_with_learning():
    """Episode should run with STDP enabled."""
    c = Connectome.from_bundled("toy20")
    runner = ClosedLoopRunner(
        c, neurons_per_region=100, global_coupling=0.05,
        enable_learning=True, action_every_ms=20.0,
        stdp_every=20, seed=42,
    )
    log = runner.run_episode()
    assert log.steps > 0


def test_closed_loop_multiple_episodes():
    """Multiple episodes should run without state leaking."""
    c = Connectome.from_bundled("toy20")
    runner = ClosedLoopRunner(
        c, neurons_per_region=100, global_coupling=0.05,
        enable_learning=False, action_every_ms=20.0, seed=42,
    )
    logs = runner.run_episodes(n_episodes=3, log_every=3)
    assert len(logs) == 3
    # Each episode should have different starting conditions
    assert logs[0].distances[0] != logs[1].distances[0] or logs[0].actions != logs[1].actions
