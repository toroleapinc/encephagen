"""Tests for Milestone 2: Learning (STDP + homeostatic plasticity)."""

import numpy as np
from scipy import sparse

from encephagen.learning.stdp import STDPRule, STDPParams
from encephagen.learning.homeostatic import HomeostaticPlasticity, HomeostaticParams


# --- STDP tests ---

def test_stdp_creates():
    stdp = STDPRule(n_pre=100, n_post=100)
    assert stdp.pre_trace.shape == (100,)
    assert stdp.post_trace.shape == (100,)


def test_stdp_traces_increment_on_spike():
    stdp = STDPRule(n_pre=10, n_post=10)
    pre_spikes = np.zeros(10, dtype=bool)
    post_spikes = np.zeros(10, dtype=bool)
    pre_spikes[0] = True
    post_spikes[5] = True

    stdp.update_traces(0.1, pre_spikes, post_spikes)
    assert stdp.pre_trace[0] > 0
    assert stdp.post_trace[5] > 0
    assert stdp.pre_trace[1] == 0  # Non-spiking neuron


def test_stdp_traces_decay():
    stdp = STDPRule(n_pre=10, n_post=10)
    pre_spikes = np.zeros(10, dtype=bool)
    post_spikes = np.zeros(10, dtype=bool)
    pre_spikes[0] = True
    stdp.update_traces(0.1, pre_spikes, post_spikes)

    val_after_spike = stdp.pre_trace[0]

    # No more spikes — should decay
    pre_spikes[0] = False
    for _ in range(500):  # 50ms > 2*tau_plus(20ms) → should decay well below 0.5
        stdp.update_traces(0.1, pre_spikes, post_spikes)

    assert stdp.pre_trace[0] < val_after_spike * 0.1


def test_stdp_ltp_pre_before_post():
    """Pre fires before post → weight should increase."""
    stdp = STDPRule(n_pre=2, n_post=2)
    weights = sparse.csr_matrix(np.array([[0.5, 0], [0, 0.5]]))

    # Pre neuron 0 fires
    pre_spikes = np.array([True, False])
    post_spikes = np.array([False, False])
    weights = stdp.step(0.1, pre_spikes, post_spikes, weights)

    # 5ms later, post neuron 0 fires (pre fired BEFORE post → LTP)
    for _ in range(50):
        pre_spikes = np.array([False, False])
        post_spikes = np.array([False, False])
        weights = stdp.step(0.1, pre_spikes, post_spikes, weights)

    post_spikes = np.array([True, False])
    pre_spikes = np.array([False, False])
    weights = stdp.step(0.1, pre_spikes, post_spikes, weights)

    # Weight 0→0 should have increased
    assert weights[0, 0] > 0.5, f"Expected LTP, got {weights[0, 0]}"


def test_stdp_ltd_post_before_pre():
    """Post fires before pre → weight should decrease."""
    stdp = STDPRule(n_pre=2, n_post=2)
    weights = sparse.csr_matrix(np.array([[1.0, 0], [0, 1.0]]))

    # Post neuron 0 fires first
    pre_spikes = np.array([False, False])
    post_spikes = np.array([True, False])
    weights = stdp.step(0.1, pre_spikes, post_spikes, weights)

    # 5ms later, pre neuron 0 fires (post fired BEFORE pre → LTD)
    for _ in range(50):
        weights = stdp.step(0.1, np.zeros(2, dtype=bool), np.zeros(2, dtype=bool), weights)

    pre_spikes = np.array([True, False])
    post_spikes = np.array([False, False])
    weights = stdp.step(0.1, pre_spikes, post_spikes, weights)

    assert weights[0, 0] < 1.0, f"Expected LTD, got {weights[0, 0]}"


def test_stdp_respects_bounds():
    """Weights should stay within [w_min, w_max]."""
    params = STDPParams(a_plus=0.5, a_minus=0.5, w_max=2.0, w_min=0.0)
    stdp = STDPRule(n_pre=2, n_post=2, params=params)
    weights = sparse.csr_matrix(np.array([[1.5, 0], [0, 0.1]]))

    # Massive LTP — should not exceed w_max
    for _ in range(100):
        pre_spikes = np.array([True, False])
        post_spikes = np.array([True, False])
        weights = stdp.step(0.1, pre_spikes, post_spikes, weights)

    assert weights[0, 0] <= params.w_max + 1e-6


def test_stdp_reset():
    stdp = STDPRule(n_pre=10, n_post=10)
    stdp.pre_trace[:] = 5.0
    stdp.post_trace[:] = 3.0
    stdp.reset()
    assert np.allclose(stdp.pre_trace, 0)
    assert np.allclose(stdp.post_trace, 0)


# --- Homeostatic plasticity tests ---

def test_homeostatic_creates():
    hp = HomeostaticPlasticity(n_neurons=100)
    assert hp.running_rate.shape == (100,)
    assert np.allclose(hp.running_rate, hp.p.target_rate)


def test_homeostatic_rate_tracking():
    """Running rate should track actual firing."""
    hp = HomeostaticPlasticity(n_neurons=10, params=HomeostaticParams(tau_homeo=100.0))

    # Simulate high firing
    spikes = np.ones(10, dtype=bool)
    for _ in range(1000):
        hp.update_rates(0.1, spikes)

    # Running rate should be high (near instantaneous rate)
    assert hp.running_rate.mean() > hp.p.target_rate


def test_homeostatic_scaling_down():
    """If firing too fast, scaling factor < 1."""
    hp = HomeostaticPlasticity(n_neurons=10)
    hp.running_rate[:] = 50.0  # Way above target (10 Hz)
    factors = hp.compute_scaling_factors()
    assert np.all(factors < 1.0), "Should scale down when firing too fast"


def test_homeostatic_scaling_up():
    """If firing too slow, scaling factor > 1."""
    hp = HomeostaticPlasticity(n_neurons=10)
    hp.running_rate[:] = 1.0  # Way below target (10 Hz)
    factors = hp.compute_scaling_factors()
    assert np.all(factors > 1.0), "Should scale up when firing too slow"


def test_homeostatic_at_target():
    """At target rate, scaling factor ≈ 1."""
    hp = HomeostaticPlasticity(n_neurons=10)
    hp.running_rate[:] = hp.p.target_rate
    factors = hp.compute_scaling_factors()
    np.testing.assert_allclose(factors, 1.0, atol=0.01)


def test_homeostatic_weight_scaling():
    """Weights should actually change after scaling."""
    hp = HomeostaticPlasticity(n_neurons=5)
    hp.running_rate[:] = 50.0  # Too fast → scale down

    weights = sparse.csr_matrix(np.array([
        [0, 1.0, 0.5, 0, 0],
        [1.0, 0, 0, 0.5, 0],
        [0, 0, 0, 1.0, 0.5],
        [0, 0, 0, 0, 1.0],
        [0.5, 0, 0, 0, 0],
    ]))

    factors = hp.compute_scaling_factors()
    scaled = hp.apply_scaling(weights, factors)

    # Weights should be smaller (scaled down)
    assert scaled.toarray().sum() < weights.toarray().sum()


def test_homeostatic_reset():
    hp = HomeostaticPlasticity(n_neurons=10)
    hp.running_rate[:] = 99.0
    hp.reset()
    np.testing.assert_allclose(hp.running_rate, hp.p.target_rate)
