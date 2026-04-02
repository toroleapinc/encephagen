"""Visual encoder: convert images to spike trains for V1/V2 regions.

Uses rate coding: pixel brightness → firing rate.
Brighter pixels produce more spikes per timestep.

Each pixel maps to a small group of neurons in the target region.
A 28x28 image maps to 784 neuron groups.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VisualParams:
    """Parameters for visual encoding."""

    max_rate: float = 100.0      # Maximum firing rate for brightest pixel (Hz)
    min_rate: float = 0.0        # Minimum rate for darkest pixel (Hz)
    gain: float = 15.0           # mV per spike (input current strength)


class VisualEncoder:
    """Encode grayscale images as spike train input to sensory regions.

    Converts a grayscale image [H, W] (values 0-1) to a per-neuron
    external current array that can be fed into a RegionPopulation.

    Each pixel is assigned to a group of neurons. The number of neurons
    per pixel group depends on the region size and image size.
    """

    def __init__(
        self,
        image_height: int = 28,
        image_width: int = 28,
        n_neurons: int = 200,
        params: VisualParams | None = None,
        seed: int | None = None,
    ):
        self.h = image_height
        self.w = image_width
        self.n_pixels = image_height * image_width
        self.n_neurons = n_neurons
        self.p = params or VisualParams()
        self.rng = np.random.default_rng(seed)

        # Assign neurons to pixels
        # If n_neurons < n_pixels: some pixels share neurons
        # If n_neurons > n_pixels: some pixels get multiple neurons
        self.pixel_to_neurons = self._build_mapping()

    def _build_mapping(self) -> list[list[int]]:
        """Map each pixel to a list of neuron indices."""
        mapping: list[list[int]] = [[] for _ in range(self.n_pixels)]

        if self.n_neurons >= self.n_pixels:
            # More neurons than pixels: distribute evenly
            neurons_per_pixel = self.n_neurons // self.n_pixels
            remainder = self.n_neurons % self.n_pixels
            idx = 0
            for p in range(self.n_pixels):
                count = neurons_per_pixel + (1 if p < remainder else 0)
                mapping[p] = list(range(idx, idx + count))
                idx += count
        else:
            # Fewer neurons than pixels: each neuron covers multiple pixels
            # Assign each neuron to the nearest pixel
            for neuron_idx in range(self.n_neurons):
                pixel_idx = neuron_idx * self.n_pixels // self.n_neurons
                mapping[pixel_idx].append(neuron_idx)

        return mapping

    def encode(self, image: np.ndarray, dt: float) -> np.ndarray:
        """Convert an image to a per-neuron external current.

        Args:
            image: Grayscale image [H, W] with values in [0, 1].
            dt: Simulation timestep in ms.

        Returns:
            External current array [n_neurons] in mV.
        """
        if image.shape != (self.h, self.w):
            raise ValueError(f"Expected image shape ({self.h}, {self.w}), got {image.shape}")

        flat = image.ravel()  # [n_pixels]
        current = np.zeros(self.n_neurons, dtype=np.float64)

        for pixel_idx in range(self.n_pixels):
            neurons = self.pixel_to_neurons[pixel_idx]
            if not neurons:
                continue

            brightness = float(np.clip(flat[pixel_idx], 0, 1))

            # Rate coding: brightness → Poisson rate → spikes → current
            rate_hz = self.p.min_rate + brightness * (self.p.max_rate - self.p.min_rate)
            rate_per_step = rate_hz * dt / 1000.0

            for n_idx in neurons:
                if self.rng.random() < rate_per_step:
                    current[n_idx] += self.p.gain

        return current

    def encode_batch(self, image: np.ndarray, dt: float, n_steps: int) -> list[np.ndarray]:
        """Encode an image as a sequence of current arrays.

        Args:
            image: Grayscale image [H, W] in [0, 1].
            dt: Timestep in ms.
            n_steps: Number of timesteps to generate.

        Returns:
            List of n_steps current arrays, each [n_neurons].
        """
        return [self.encode(image, dt) for _ in range(n_steps)]

    def decode_rates(self, spike_counts: np.ndarray, duration_ms: float) -> np.ndarray:
        """Decode neuron spike counts back to an approximate image.

        Args:
            spike_counts: Total spikes per neuron [n_neurons] over duration.
            duration_ms: Duration in ms.

        Returns:
            Reconstructed image [H, W] in [0, 1].
        """
        image = np.zeros(self.n_pixels, dtype=np.float64)
        duration_s = duration_ms / 1000.0

        for pixel_idx in range(self.n_pixels):
            neurons = self.pixel_to_neurons[pixel_idx]
            if not neurons:
                continue
            # Average firing rate of assigned neurons
            rate = np.mean(spike_counts[neurons]) / duration_s
            # Inverse rate coding
            brightness = (rate - self.p.min_rate) / (self.p.max_rate - self.p.min_rate + 1e-12)
            image[pixel_idx] = np.clip(brightness, 0, 1)

        return image.reshape(self.h, self.w)
