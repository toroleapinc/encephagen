"""Tests for Milestone 3: Sensory input encoding."""

import numpy as np

from encephagen.sensory.visual import VisualEncoder, VisualParams
from encephagen.sensory.auditory import AuditoryEncoder, AuditoryParams


# --- Visual encoder tests ---

def test_visual_creates():
    enc = VisualEncoder(image_height=28, image_width=28, n_neurons=200, seed=42)
    assert enc.n_pixels == 784
    assert enc.n_neurons == 200


def test_visual_encode_shape():
    enc = VisualEncoder(image_height=28, image_width=28, n_neurons=200, seed=42)
    image = np.random.rand(28, 28)
    current = enc.encode(image, dt=0.1)
    assert current.shape == (200,)


def test_visual_black_image_low_current():
    """Black image (all zeros) should produce minimal current."""
    enc = VisualEncoder(n_neurons=200, seed=42)
    black = np.zeros((28, 28))
    currents = [enc.encode(black, dt=0.1) for _ in range(100)]
    mean_current = np.mean([c.sum() for c in currents])
    assert mean_current < 5.0, f"Black image produced too much current: {mean_current}"


def test_visual_white_image_high_current():
    """White image (all ones) should produce strong current."""
    enc = VisualEncoder(n_neurons=200, seed=42)
    white = np.ones((28, 28))
    currents = [enc.encode(white, dt=0.1) for _ in range(100)]
    mean_current = np.mean([c.sum() for c in currents])
    assert mean_current > 0, "White image produced zero current"


def test_visual_brightness_monotonic():
    """Brighter images should produce more current on average."""
    enc = VisualEncoder(n_neurons=200, seed=42)

    dark = np.full((28, 28), 0.2)
    bright = np.full((28, 28), 0.8)

    dark_total = sum(enc.encode(dark, dt=0.1).sum() for _ in range(200))
    bright_total = sum(enc.encode(bright, dt=0.1).sum() for _ in range(200))

    assert bright_total > dark_total, "Brighter image should produce more current"


def test_visual_encode_batch():
    enc = VisualEncoder(n_neurons=200, seed=42)
    image = np.random.rand(28, 28)
    batch = enc.encode_batch(image, dt=0.1, n_steps=10)
    assert len(batch) == 10
    assert all(c.shape == (200,) for c in batch)


def test_visual_decode_roundtrip():
    """Encode then decode should roughly preserve the image."""
    enc = VisualEncoder(n_neurons=784, seed=42)  # 1 neuron per pixel
    image = np.random.rand(28, 28)

    # Accumulate spikes over many timesteps
    spike_counts = np.zeros(784)
    n_steps = 5000
    for _ in range(n_steps):
        current = enc.encode(image, dt=0.1)
        spike_counts += (current > 0).astype(float)

    decoded = enc.decode_rates(spike_counts, duration_ms=n_steps * 0.1)
    assert decoded.shape == (28, 28)

    # Correlation should be positive (not perfect due to Poisson noise)
    corr = np.corrcoef(image.ravel(), decoded.ravel())[0, 1]
    assert corr > 0.3, f"Decoded image poorly correlated with original: {corr:.3f}"


def test_visual_wrong_image_shape_raises():
    enc = VisualEncoder(image_height=28, image_width=28, n_neurons=200)
    import pytest
    with pytest.raises(ValueError, match="Expected image shape"):
        enc.encode(np.zeros((32, 32)), dt=0.1)


# --- Auditory encoder tests ---

def test_auditory_creates():
    enc = AuditoryEncoder(n_neurons=200, seed=42)
    assert enc.n_neurons == 200
    assert len(enc.band_to_neurons) == enc.p.n_bands


def test_auditory_encode_shape():
    enc = AuditoryEncoder(n_neurons=200, seed=42)
    enc.encode_tone(440.0, amplitude=1.0)
    current = enc.encode(dt=0.1)
    assert current.shape == (200,)


def test_auditory_tone_activates_correct_band():
    """A 440Hz tone should activate the band containing 440Hz."""
    enc = AuditoryEncoder(n_neurons=200, seed=42)
    enc.encode_tone(440.0, amplitude=1.0)

    # Find which band 440Hz falls in
    target_band = None
    for b in range(enc.p.n_bands):
        if enc.band_edges[b] <= 440 < enc.band_edges[b + 1]:
            target_band = b
            break

    assert target_band is not None
    assert enc._current_spectrum[target_band] == 1.0

    # Other bands should be zero
    for b in range(enc.p.n_bands):
        if b != target_band:
            assert enc._current_spectrum[b] == 0.0


def test_auditory_silence_no_current():
    """No tone (all zeros spectrum) should produce minimal current."""
    enc = AuditoryEncoder(n_neurons=200, seed=42)
    enc._current_spectrum[:] = 0
    currents = [enc.encode(dt=0.1) for _ in range(100)]
    total = sum(c.sum() for c in currents)
    assert total == 0, "Silence produced current"


def test_auditory_analyze_window():
    """Analyze a sine wave window."""
    enc = AuditoryEncoder(n_neurons=200, seed=42)
    sr = enc.p.sample_rate
    t = np.arange(int(enc.p.window_ms * sr / 1000)) / sr
    tone_440 = np.sin(2 * np.pi * 440 * t)
    enc.analyze_window(tone_440)

    # The spectrum should have energy
    assert enc._current_spectrum.max() > 0
