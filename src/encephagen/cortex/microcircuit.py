"""Canonical Cortical Microcircuit — the genetic recipe for cortex.

Every cortical region uses the same basic circuit (Douglas & Martin 1991,
Harris & Shepherd 2015). This module defines that circuit and generates
structured connectivity for each region.

Cell types (from Allen Brain Atlas / MICrONS):
  Excitatory:
    L4_stellate  — receives thalamic/feedforward input
    L23_pyramid  — local processing, sends feedforward to other regions
    L5_pyramid   — main output to subcortex, brainstem, spinal cord
    L6_pyramid   — corticothalamic feedback

  Inhibitory:
    PV_basket    — fast-spiking, perisomatic inhibition → timing
    SST_martinotti — slow, dendritic inhibition → gain control
    VIP          — inhibits SST → disinhibition (gates learning)

Connectivity rules (from MICrONS, Blue Brain Project, Markram 2015):
  L4 → L2/3 (strong feedforward)
  L2/3 → L5 (cortical output pathway)
  L5 → L6 (output copy)
  L6 → L4 (intracortical feedback)
  PV → all excitatory (perisomatic, strong)
  SST → L23/L5 dendrites (weaker, sustained)
  VIP → SST (disinhibition)
  All excitatory → PV, SST (drive inhibition)

Between-region rules:
  FEEDFORWARD: L2/3 pyramidal → target region L4 (driving, strong)
  FEEDBACK: L5/6 pyramidal → target region L2/3 (modulatory, weak)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class CellTypeConfig:
    """Configuration for cell types in the canonical microcircuit."""
    # Neurons per region (200 total)
    n_L4_stellate: int = 25
    n_L23_pyramid: int = 45
    n_L5_pyramid: int = 25
    n_L6_pyramid: int = 15
    n_PV_basket: int = 30      # ~15% of cortical neurons
    n_SST: int = 20            # ~10%
    n_VIP: int = 10            # ~5%
    # Total: 170 assigned + 30 buffer = 200

    # Time constants per cell type (from electrophysiology)
    tau_L4: float = 15.0       # fast (receives thalamic input)
    tau_L23: float = 20.0      # standard
    tau_L5: float = 25.0       # slower (integrates more)
    tau_L6: float = 20.0       # standard
    tau_PV: float = 8.0        # FAST spiking (defining feature of PV+)
    tau_SST: float = 15.0      # moderate
    tau_VIP: float = 12.0      # moderate-fast


def build_microcircuit_connectivity(
    npr: int = 200,
    n_regions: int = 80,
    config: CellTypeConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build internal connectivity for all regions using canonical microcircuit.

    Returns:
        W_internal: sparse arrays (rows, cols, vals) for within-region connections
        tau_m_array: per-neuron membrane time constant
        cell_map: dict mapping (region, cell_type) → neuron indices
    """
    if config is None:
        config = CellTypeConfig()
    if rng is None:
        rng = np.random.default_rng(42)

    N = n_regions * npr
    rows, cols, vals = [], [], []
    tau_m = np.full(N, 20.0, dtype=np.float32)
    cell_map = {}

    # Connection probabilities (from MICrONS / Blue Brain Project literature)
    # Format: (source_type, target_type, probability, weight, description)
    RULES = [
        # Feedforward pathway: L4 → L2/3 → L5 → L6
        ('L4', 'L23', 0.15, 1.2, 'L4 drives L2/3'),
        ('L23', 'L5', 0.10, 1.0, 'L2/3 drives L5 output'),
        ('L5', 'L6', 0.08, 0.8, 'output copy to L6'),
        ('L6', 'L4', 0.05, 0.6, 'intracortical feedback'),

        # Recurrent excitation (within layer)
        ('L23', 'L23', 0.08, 0.5, 'L2/3 recurrent'),
        ('L5', 'L5', 0.06, 0.5, 'L5 recurrent'),

        # Excitatory → Inhibitory (drives inhibition)
        ('L23', 'PV', 0.20, 1.0, 'L2/3 drives PV'),
        ('L23', 'SST', 0.15, 0.8, 'L2/3 drives SST'),
        ('L5', 'PV', 0.15, 0.8, 'L5 drives PV'),
        ('L4', 'PV', 0.12, 0.8, 'L4 drives PV'),

        # PV inhibition (fast, perisomatic — controls TIMING)
        ('PV', 'L23', 0.30, -2.5, 'PV inhibits L2/3'),
        ('PV', 'L5', 0.25, -2.0, 'PV inhibits L5'),
        ('PV', 'L4', 0.20, -1.5, 'PV inhibits L4'),

        # SST inhibition (slow, dendritic — controls GAIN)
        ('SST', 'L23', 0.20, -1.5, 'SST inhibits L2/3 dendrites'),
        ('SST', 'L5', 0.15, -1.2, 'SST inhibits L5 dendrites'),

        # VIP disinhibition (VIP → SST, releasing excitatory neurons)
        ('VIP', 'SST', 0.25, -2.0, 'VIP inhibits SST → disinhibition'),

        # Excitatory → VIP (attention/arousal signal activates VIP)
        ('L23', 'VIP', 0.10, 0.6, 'L2/3 drives VIP'),
    ]

    # Cell type sizes and tau
    type_config = {
        'L4': (config.n_L4_stellate, config.tau_L4),
        'L23': (config.n_L23_pyramid, config.tau_L23),
        'L5': (config.n_L5_pyramid, config.tau_L5),
        'L6': (config.n_L6_pyramid, config.tau_L6),
        'PV': (config.n_PV_basket, config.tau_PV),
        'SST': (config.n_SST, config.tau_SST),
        'VIP': (config.n_VIP, config.tau_VIP),
    }

    for r in range(n_regions):
        offset = r * npr
        neuron_idx = offset

        # Assign neuron indices to cell types
        region_map = {}
        for ctype, (count, tau) in type_config.items():
            start = neuron_idx
            end = min(neuron_idx + count, offset + npr)
            region_map[ctype] = (start, end)
            tau_m[start:end] = tau
            cell_map[(r, ctype)] = (start, end)
            neuron_idx = end

        # Fill remaining neurons as L23 (most common type)
        if neuron_idx < offset + npr:
            s, e = region_map['L23']
            region_map['L23'] = (s, offset + npr)
            tau_m[e:offset + npr] = config.tau_L23
            cell_map[(r, 'L23')] = (s, offset + npr)

        # Apply connectivity rules
        for src_type, dst_type, prob, weight, desc in RULES:
            src_start, src_end = region_map[src_type]
            dst_start, dst_end = region_map[dst_type]
            n_src = src_end - src_start
            n_dst = dst_end - dst_start

            if n_src == 0 or n_dst == 0:
                continue

            # Generate connections
            for i in range(src_start, src_end):
                for j in range(dst_start, dst_end):
                    if i != j and rng.random() < prob:
                        rows.append(i)
                        cols.append(j)
                        vals.append(weight / max(n_src, 1))

    return (np.array(rows), np.array(cols), np.array(vals, dtype=np.float32)), tau_m, cell_map


def build_between_region_connectivity(
    connectome_weights: np.ndarray,
    cell_map: dict,
    n_regions: int,
    npr: int,
    global_coupling: float = 5.0,
    rng: np.random.Generator | None = None,
    g_contra_ratio: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build between-region connections following feedforward/feedback rules.

    FEEDFORWARD: L2/3 → target L4 (strong, driving)
    FEEDBACK: L5/6 → target L2/3 (weak, modulatory)

    The distinction is based on hierarchical position from T1w/T2w:
      Lower T1w/T2w (sensory, faster) → Higher T1w/T2w (frontal, slower) = FEEDFORWARD
      Higher → Lower = FEEDBACK
    """
    if rng is None:
        rng = np.random.default_rng(42)

    rows, cols, vals = [], [], []

    # Normalize weights
    all_w = connectome_weights[connectome_weights > 0]
    if len(all_w) == 0:
        return np.array([]), np.array([]), np.array([], dtype=np.float32)

    dyn_range = all_w.max() / all_w.min()
    if dyn_range > 100:
        log_max = np.log1p(all_w.max() * 1000)
        use_log = True
    else:
        use_log = False
        w_max = all_w.max()

    # Load T1w/T2w for hierarchy (lower tau = sensory = lower in hierarchy)
    try:
        tau_m_regions = np.load(
            'src/encephagen/connectome/bundled/neurolib80_tau_m.npy')
    except FileNotFoundError:
        tau_m_regions = np.linspace(10, 30, n_regions)

    # Hemisphere-specific coupling (2025 literature: improves FC-FC)
    # Intra-hemispheric connections are stronger than inter-hemispheric
    try:
        tau_labels_loaded = True
        import json
        _tau_labels = json.load(open(
            'src/encephagen/connectome/bundled/neurolib80_t1t2_labels.json'))
    except (FileNotFoundError, Exception):
        _tau_labels = [f'region_{i}' for i in range(n_regions)]
        tau_labels_loaded = False

    def same_hemisphere(src_r, dst_r):
        if not tau_labels_loaded or src_r >= len(_tau_labels) or dst_r >= len(_tau_labels):
            return True
        s = _tau_labels[src_r]; d = _tau_labels[dst_r]
        s_hemi = 'L' if '_L' in s else ('R' if '_R' in s else '?')
        d_hemi = 'L' if '_L' in d else ('R' if '_R' in d else '?')
        return s_hemi == d_hemi

    n_ff = 0
    n_fb = 0

    for src in range(n_regions):
        for dst in range(n_regions):
            if src == dst:
                continue
            w = connectome_weights[src, dst]
            if w <= 0:
                continue

            # Normalize weight
            if use_log:
                w_norm = np.log1p(w * 1000) / log_max
            else:
                w_norm = w / w_max

            # Determine feedforward vs feedback from hierarchy
            # Lower tau_m (sensory) → higher tau_m (frontal) = feedforward
            is_feedforward = tau_m_regions[src] < tau_m_regions[dst]

            # Hemisphere-specific scaling
            hemi_scale = 1.0 if same_hemisphere(src, dst) else g_contra_ratio

            if is_feedforward:
                # FEEDFORWARD: L2/3 of source → L4 of target (strong, driving)
                src_start, src_end = cell_map.get((src, 'L23'), (0, 0))
                dst_start, dst_end = cell_map.get((dst, 'L4'), (0, 0))
                syn_weight = global_coupling * w_norm * 1.5 * hemi_scale
                conn_prob = 0.03 * w_norm
                n_ff += 1
            else:
                # FEEDBACK: L5/6 of source → L2/3 of target (weak, modulatory)
                src_s5, src_e5 = cell_map.get((src, 'L5'), (0, 0))
                src_s6, src_e6 = cell_map.get((src, 'L6'), (0, 0))
                src_start = src_s5
                src_end = src_e6 if src_e6 > 0 else src_e5
                dst_start, dst_end = cell_map.get((dst, 'L23'), (0, 0))
                syn_weight = global_coupling * w_norm * 0.5 * hemi_scale  # weak
                conn_prob = 0.02 * w_norm
                n_fb += 1

            n_src = src_end - src_start
            if n_src == 0 or dst_end - dst_start == 0:
                continue

            # Feedforward inhibition: also connect to PV in target (creates E-I sequence)
            dst_pv_start, dst_pv_end = cell_map.get((dst, 'PV'), (0, 0))

            for i in range(src_start, src_end):
                # Connect to target excitatory
                for j in range(dst_start, dst_end):
                    if rng.random() < conn_prob:
                        rows.append(i)
                        cols.append(j)
                        vals.append(syn_weight / max(n_src, 1))

                # Feedforward inhibition: stronger connection to target PV
                if dst_pv_end > dst_pv_start:
                    for j in range(dst_pv_start, dst_pv_end):
                        if rng.random() < conn_prob * 1.5:  # PV gets MORE input
                            rows.append(i)
                            cols.append(j)
                            vals.append(syn_weight * 1.5 / max(n_src, 1))

    print(f"  Microcircuit: {n_ff} feedforward + {n_fb} feedback pathways")
    return np.array(rows), np.array(cols), np.array(vals, dtype=np.float32)
