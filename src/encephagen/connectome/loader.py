"""Load and represent brain connectivity graphs."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch


_BUNDLED_DIR = Path(__file__).parent / "bundled"


class Connectome:
    """A brain connectivity graph: regions (nodes) and weighted connections (edges)."""

    def __init__(
        self,
        weights: np.ndarray,
        labels: list[str],
        positions: np.ndarray | None = None,
        region_types: dict[str, str] | None = None,
    ):
        if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
            raise ValueError(
                f"weights must be a square 2D array, got shape {weights.shape}"
            )
        if len(labels) != weights.shape[0]:
            raise ValueError(
                f"labels length {len(labels)} != matrix size {weights.shape[0]}"
            )
        if np.isnan(weights).any() or np.isinf(weights).any():
            raise ValueError("weights contain NaN or Inf values")
        if (weights < 0).any():
            import warnings
            warnings.warn(
                f"Connectome weights contain {int((weights < 0).sum())} negative "
                f"values. These will be treated as absent connections.",
                UserWarning,
                stacklevel=2,
            )
        self.weights = weights.astype(np.float32)
        self.labels = list(labels)
        self.positions = positions
        self.region_types = region_types or {}

    @classmethod
    def from_bundled(cls, name: str = "toy20") -> Connectome:
        """Load a bundled connectivity matrix by name."""
        if name == "toy20":
            return _make_toy_connectome()
        weights_path = _BUNDLED_DIR / f"{name}_weights.npy"
        labels_path = _BUNDLED_DIR / f"{name}_labels.json"
        positions_path = _BUNDLED_DIR / f"{name}_positions.npy"
        return cls.from_files(
            str(weights_path),
            str(labels_path),
            str(positions_path) if positions_path.exists() else None,
        )

    @classmethod
    def from_files(
        cls,
        weights_path: str,
        labels_path: str,
        positions_path: str | None = None,
    ) -> Connectome:
        """Load from user-provided files."""
        weights = np.load(weights_path)
        with open(labels_path, encoding="utf-8") as f:
            labels = json.load(f)
        positions = np.load(positions_path) if positions_path else None
        return cls(weights, labels, positions)

    @classmethod
    def from_numpy(cls, weights: np.ndarray, labels: list[str] | None = None) -> Connectome:
        """Create from a raw NumPy adjacency matrix."""
        n = weights.shape[0]
        if labels is None:
            labels = [f"region_{i}" for i in range(n)]
        return cls(weights, labels)

    @property
    def num_regions(self) -> int:
        return len(self.labels)

    @property
    def adjacency(self) -> np.ndarray:
        """Binary adjacency matrix (nonzero entries)."""
        return (self.weights > 0).astype(np.float32)

    @property
    def num_edges(self) -> int:
        return int((self.weights > 0).sum())

    def get_neighbors(self, region_idx: int) -> list[int]:
        """Get outgoing neighbors of a region."""
        return [int(x) for x in np.where(self.weights[region_idx] > 0)[0]]

    def get_incoming(self, region_idx: int) -> list[int]:
        """Get regions that project TO this region."""
        return [int(x) for x in np.where(self.weights[:, region_idx] > 0)[0]]

    def get_weight(self, src: int, dst: int) -> float:
        return float(self.weights[src, dst])

    def edges(self) -> list[tuple[int, int, float]]:
        """All edges as (src, dst, weight) tuples."""
        rows, cols = np.where(self.weights > 0)
        return [(int(r), int(c), float(self.weights[r, c])) for r, c in zip(rows, cols)]

    def to_edge_index(self) -> torch.LongTensor:
        """Convert to PyG-style edge_index tensor [2, num_edges]."""
        rows, cols = np.where(self.weights > 0)
        return torch.tensor(np.stack([rows, cols]), dtype=torch.long)

    def to_edge_weights(self) -> torch.FloatTensor:
        """Edge weights corresponding to to_edge_index()."""
        rows, cols = np.where(self.weights > 0)
        return torch.tensor(
            [self.weights[r, c] for r, c in zip(rows, cols)], dtype=torch.float32
        )

    def to_networkx(self) -> nx.DiGraph:
        """Convert to a NetworkX directed graph."""
        g = nx.DiGraph()
        for i, label in enumerate(self.labels):
            g.add_node(i, label=label)
        for src, dst, w in self.edges():
            g.add_edge(src, dst, weight=w)
        return g

    def __repr__(self) -> str:
        return (
            f"Connectome(regions={self.num_regions}, "
            f"edges={self.num_edges}, "
            f"density={self.num_edges / (self.num_regions ** 2):.2%})"
        )


def _make_toy_connectome() -> Connectome:
    """Create a toy 20-region connectome loosely inspired by human brain organization.

    Regions are grouped into functional clusters (visual, motor, prefrontal,
    temporal, subcortical) with denser intra-cluster connectivity and sparser
    inter-cluster connections mimicking real brain wiring patterns.
    """
    labels = [
        # Visual (0-3)
        "V1_left", "V1_right", "V2_left", "V2_right",
        # Motor (4-7)
        "M1_left", "M1_right", "SMA_left", "SMA_right",
        # Prefrontal (8-11)
        "dlPFC_left", "dlPFC_right", "vmPFC_left", "vmPFC_right",
        # Temporal (12-15)
        "A1_left", "A1_right", "Wernicke_left", "Broca_left",
        # Subcortical (16-19)
        "Thalamus_left", "Thalamus_right", "Hippocampus_left", "Hippocampus_right",
    ]

    n = len(labels)
    rng = np.random.default_rng(42)
    w = np.zeros((n, n), dtype=np.float32)

    # Intra-cluster connections (dense)
    clusters = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20)]
    for start, end in clusters:
        for i in range(start, end):
            for j in range(start, end):
                if i != j:
                    w[i, j] = rng.uniform(0.5, 1.0)

    # Inter-cluster connections (sparse, biologically motivated)
    inter = [
        # Visual -> Temporal (ventral stream)
        (0, 12), (1, 13), (2, 14),
        # Visual -> Motor (dorsal stream)
        (0, 4), (1, 5), (2, 6),
        # Motor <-> Prefrontal (executive control)
        (4, 8), (5, 9), (8, 4), (9, 5),
        # Temporal -> Prefrontal (language pathway)
        (14, 15), (15, 8), (15, 9),
        # Thalamus <-> everything (relay)
        *[(16, i) for i in range(16) if i % 2 == 0],
        *[(17, i) for i in range(16) if i % 2 == 1],
        *[(i, 16) for i in range(16) if i % 2 == 0],
        *[(i, 17) for i in range(16) if i % 2 == 1],
        # Hippocampus <-> Prefrontal (memory-executive)
        (18, 8), (18, 10), (19, 9), (19, 11),
        (8, 18), (10, 18), (9, 19), (11, 19),
        # Hippocampus <-> Temporal (memory encoding)
        (18, 12), (18, 14), (19, 13),
        (12, 18), (14, 18), (13, 19),
    ]
    for src, dst in inter:
        if src < n and dst < n:
            w[src, dst] = rng.uniform(0.2, 0.6)

    # Normalize weights to [0, 1]
    if w.max() > 0:
        w /= w.max()

    return Connectome(
        weights=w,
        labels=labels,
        region_types={
            **{l: "thalamus" for l in labels if "Thalamus" in l},
            **{l: "hippocampus" for l in labels if "Hippocampus" in l},
        },
    )
