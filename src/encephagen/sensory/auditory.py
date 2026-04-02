"""Auditory encoder: convert audio signals to spike trains for A1/A2 regions.

Uses frequency-to-place coding (like the cochlea):
- Apply FFT to short windows
- Each frequency band maps to a group of neurons
- Amplitude → firing rate

Each frequency band is assigned to a group of neurons in the target region.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AuditoryParams:
    """Parameters for auditory encoding."""

    n_bands: int = 64            # Number of frequency bands
    max_rate: float = 100.0      # Max firing rate (Hz)
    min_rate: float = 0.0        # Min firing rate (Hz)
    gain: float = 15.0           # mV per spike
    window_ms: float = 25.0      # FFT window size (ms)
    sample_rate: float = 16000.0 # Audio sample rate (Hz)
    freq_min: float = 80.0       # Minimum frequency (Hz)
    freq_max: float = 8000.0     # Maximum frequency (Hz)


class AuditoryEncoder:
    """Encode audio waveforms as spike train input to auditory regions.

    Applies short-time FFT and maps frequency bands to neuron groups
    using log-spaced frequency bins (like the cochlea).
    """

    def __init__(
        self,
        n_neurons: int = 200,
        params: AuditoryParams | None = None,
        seed: int | None = None,
    ):
        self.n_neurons = n_neurons
        self.p = params or AuditoryParams()
        self.rng = np.random.default_rng(seed)

        # Log-spaced frequency band edges (like cochlea)
        self.band_edges = np.logspace(
            np.log10(self.p.freq_min),
            np.log10(self.p.freq_max),
            self.p.n_bands + 1,
        )

        # Map each frequency band to neuron group
        self.band_to_neurons = self._build_mapping()

        # Internal buffer for windowed FFT
        window_samples = int(self.p.window_ms * self.p.sample_rate / 1000)
        self._window_size = window_samples
        self._current_spectrum = np.zeros(self.p.n_bands, dtype=np.float64)

    def _build_mapping(self) -> list[list[int]]:
        """Map each frequency band to neuron indices."""
        mapping: list[list[int]] = [[] for _ in range(self.p.n_bands)]
        neurons_per_band = max(1, self.n_neurons // self.p.n_bands)
        idx = 0
        for b in range(self.p.n_bands):
            count = min(neurons_per_band, self.n_neurons - idx)
            if count <= 0:
                break
            mapping[b] = list(range(idx, idx + count))
            idx += count
        # Assign remaining neurons to last band
        if idx < self.n_neurons:
            mapping[-1].extend(range(idx, self.n_neurons))
        return mapping

    def analyze_window(self, audio_window: np.ndarray) -> None:
        """Analyze an audio window to update the current spectrum.

        Args:
            audio_window: Audio samples for one window. Values in [-1, 1].
        """
        # Zero-pad or truncate to window size
        if len(audio_window) < self._window_size:
            padded = np.zeros(self._window_size)
            padded[:len(audio_window)] = audio_window
            audio_window = padded
        elif len(audio_window) > self._window_size:
            audio_window = audio_window[:self._window_size]

        # FFT
        spectrum = np.abs(np.fft.rfft(audio_window))
        freqs = np.fft.rfftfreq(self._window_size, d=1.0 / self.p.sample_rate)

        # Bin into frequency bands
        for b in range(self.p.n_bands):
            low = self.band_edges[b]
            high = self.band_edges[b + 1]
            mask = (freqs >= low) & (freqs < high)
            if mask.any():
                self._current_spectrum[b] = float(np.mean(spectrum[mask]))
            else:
                self._current_spectrum[b] = 0.0

        # Normalize to [0, 1]
        max_val = self._current_spectrum.max()
        if max_val > 0:
            self._current_spectrum /= max_val

    def encode(self, dt: float) -> np.ndarray:
        """Generate spike-based current from current spectrum.

        Call analyze_window() first to set the spectrum,
        then call encode() each timestep.

        Args:
            dt: Simulation timestep in ms.

        Returns:
            External current array [n_neurons] in mV.
        """
        current = np.zeros(self.n_neurons, dtype=np.float64)

        for band_idx in range(self.p.n_bands):
            neurons = self.band_to_neurons[band_idx]
            if not neurons:
                continue

            amplitude = self._current_spectrum[band_idx]
            rate_hz = self.p.min_rate + amplitude * (self.p.max_rate - self.p.min_rate)
            rate_per_step = rate_hz * dt / 1000.0

            for n_idx in neurons:
                if self.rng.random() < rate_per_step:
                    current[n_idx] += self.p.gain

        return current

    def encode_tone(self, frequency: float, amplitude: float = 1.0) -> None:
        """Set the spectrum to a pure tone (for testing).

        Args:
            frequency: Tone frequency in Hz.
            amplitude: Amplitude 0-1.
        """
        self._current_spectrum[:] = 0
        for b in range(self.p.n_bands):
            low = self.band_edges[b]
            high = self.band_edges[b + 1]
            if low <= frequency < high:
                self._current_spectrum[b] = amplitude
                break
