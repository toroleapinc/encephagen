"""Generate all figures for encephagen.

Figure 1: Phase transition — silencing order across coupling strengths
Figure 2: Functional hierarchy — all 96 regions ranked by variance
Figure 3: What specific wiring contributes — FC patterns real vs null
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import stats

from encephagen.connectome import Connectome
from encephagen.dynamics.brain_sim import BrainSimulator
from encephagen.dynamics.wilson_cowan import WilsonCowanParams
from encephagen.analysis.functional_roles import (
    compute_regional_profiles,
    _classify_tvb76_regions,
)

plt.style.use("default")
mpl.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
    "savefig.facecolor": "#FAFAFA",
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

GROUP_COLORS = {
    "basal_ganglia": "#e63946",
    "prefrontal": "#457b9d",
    "thalamus": "#2a9d8f",
    "hippocampus": "#e9c46a",
    "sensory": "#f4a261",
    "motor": "#264653",
    "other": "#adb5bd",
}

GROUP_LABELS = {
    "basal_ganglia": "Basal Ganglia",
    "prefrontal": "Prefrontal",
    "thalamus": "Thalamus",
    "hippocampus": "Hippocampus",
    "sensory": "Sensory",
    "motor": "Motor",
    "other": "Other Cortical",
}


def _params():
    return WilsonCowanParams(
        w_ee=16.0, w_ei=12.0, w_ie=15.0, w_ii=3.0,
        theta_e=2.0, theta_i=3.7, a_e=1.5, a_i=1.0,
        noise_sigma=0.01,
    )


def figure1_phase_transition():
    """Phase transition: which region types silence first as coupling increases."""
    print("Generating Figure 1: Phase transition...")

    connectome = Connectome.from_bundled("tvb96")
    groups = _classify_tvb76_regions(connectome.labels)
    params = _params()

    couplings = [0.001, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]
    plot_groups = ["basal_ganglia", "prefrontal", "thalamus", "hippocampus", "sensory", "motor"]

    group_traces = {g: [] for g in plot_groups}

    for G in couplings:
        sim = BrainSimulator(connectome, global_coupling=G, params=params)
        r = sim.simulate(duration=3000, dt=0.1, transient=500, seed=42)
        for gname in plot_groups:
            indices = groups.get(gname, [])
            if indices:
                mv = float(np.mean([np.var(r.E[:, i]) for i in indices]))
                group_traces[gname].append(mv)
            else:
                group_traces[gname].append(0)

    fig, ax = plt.subplots(figsize=(10, 7))

    for gname in plot_groups:
        ax.plot(couplings, group_traces[gname],
                color=GROUP_COLORS[gname], linewidth=2.5,
                marker="o", markersize=7, markerfacecolor="white",
                markeredgewidth=2, markeredgecolor=GROUP_COLORS[gname],
                label=GROUP_LABELS[gname])

    # Silencing threshold line
    ax.axhline(0.001, color="#999999", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(0.11, 0.0015, "silencing threshold", fontsize=9, color="#999999")

    ax.set_xlabel("Global Coupling Strength (G)")
    ax.set_ylabel("Mean Variance (oscillation strength)")
    ax.set_xscale("log")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="center right", frameon=True, fancybox=False,
              edgecolor="#CCCCCC", fontsize=10)

    # Annotations
    ax.annotate("Basal ganglia\nnever silences",
                xy=(0.1, group_traces["basal_ganglia"][-1]),
                xytext=(0.045, 0.155),
                fontsize=10, fontweight="bold", color=GROUP_COLORS["basal_ganglia"],
                arrowprops=dict(arrowstyle="-|>", color=GROUP_COLORS["basal_ganglia"], lw=1.5),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=GROUP_COLORS["basal_ganglia"], linewidth=1.5))

    ax.annotate("Thalamus + Motor\nsilence first",
                xy=(0.015, 0.001),
                xytext=(0.003, 0.04),
                fontsize=10, fontweight="bold", color=GROUP_COLORS["thalamus"],
                arrowprops=dict(arrowstyle="-|>", color=GROUP_COLORS["thalamus"], lw=1.5),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=GROUP_COLORS["thalamus"], linewidth=1.5))

    fig.text(0.5, 0.97,
             "Emergent Functional Hierarchy From Identical Parameters",
             ha="center", fontsize=15, fontweight="bold", color="#1a1a2e")
    fig.text(0.5, 0.925,
             "Wilson-Cowan dynamics on 96-region human connectome  |  All regions have identical local parameters",
             ha="center", fontsize=10, color="#777777")

    ax.set_position([0.1, 0.08, 0.85, 0.80])

    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/fig1_phase_transition.png")
    fig.savefig("figures/fig1_phase_transition.svg")
    plt.close()
    print("  Saved: figures/fig1_phase_transition.png")


def figure2_hierarchy_bar():
    """All 96 regions ranked by variance, colored by group."""
    print("Generating Figure 2: Regional hierarchy...")

    connectome = Connectome.from_bundled("tvb96")
    groups = _classify_tvb76_regions(connectome.labels)
    params = _params()

    sim = BrainSimulator(connectome, global_coupling=0.03, params=params)
    r = sim.simulate(duration=5000, dt=0.1, transient=1000, seed=42)
    profiles = compute_regional_profiles(r)

    # Build sorted list
    rows = []
    for label, p in profiles.items():
        idx = p["index"]
        group = "other"
        for g, indices in groups.items():
            if idx in indices:
                group = g
                break
        rows.append({"label": label, "group": group, "variance": p["variance"]})

    rows.sort(key=lambda r: -r["variance"])

    fig, ax = plt.subplots(figsize=(16, 6))

    x = np.arange(len(rows))
    colors = [GROUP_COLORS[r["group"]] for r in rows]
    bars = ax.bar(x, [r["variance"] for r in rows], color=colors, width=0.8, edgecolor="none")

    # Label top regions
    for i, r in enumerate(rows[:8]):
        ax.text(i, r["variance"] + 0.003, r["label"],
                ha="center", va="bottom", fontsize=7, rotation=45, fontweight="bold")

    ax.set_ylabel("Variance (oscillation strength)")
    ax.set_xlabel("Regions (ranked by variance)")
    ax.set_xlim(-1, len(rows))
    ax.grid(True, axis="y", alpha=0.2)

    # Remove x ticks (too many labels)
    ax.set_xticks([])

    # Legend
    legend_handles = [mpl.patches.Patch(facecolor=GROUP_COLORS[g], label=GROUP_LABELS[g])
                      for g in ["basal_ganglia", "prefrontal", "thalamus",
                                "hippocampus", "sensory", "motor", "other"]]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True,
              fancybox=False, edgecolor="#CCCCCC", fontsize=9, ncol=2)

    # Tier annotations
    ax.axhline(0.1, color="#999999", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(len(rows) - 2, 0.105, "oscillating", fontsize=8, color="#999999", ha="right")
    ax.axhline(0.01, color="#999999", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(len(rows) - 2, 0.013, "moderate", fontsize=8, color="#999999", ha="right")

    fig.text(0.5, 0.97,
             "All 96 Brain Regions Ranked by Emergent Dynamical Activity",
             ha="center", fontsize=15, fontweight="bold", color="#1a1a2e")
    fig.text(0.5, 0.925,
             "G=0.03  |  Identical Wilson-Cowan parameters everywhere  |  Differentiation is purely from connectivity position",
             ha="center", fontsize=10, color="#777777")

    ax.set_position([0.07, 0.10, 0.90, 0.78])

    fig.savefig("figures/fig2_hierarchy_bar.png")
    fig.savefig("figures/fig2_hierarchy_bar.svg")
    plt.close()
    print("  Saved: figures/fig2_hierarchy_bar.png")


def figure3_fc_comparison():
    """What specific wiring contributes: FC heatmaps real vs degree-preserving."""
    print("Generating Figure 3: FC comparison...")

    connectome = Connectome.from_bundled("tvb96")
    groups = _classify_tvb76_regions(connectome.labels)
    params = _params()
    G = 0.01

    # Real FC
    sim = BrainSimulator(connectome, global_coupling=G, params=params)
    r = sim.simulate(duration=5000, dt=0.1, transient=1000, seed=42)
    real_fc = np.corrcoef(r.E.T)

    # Degree-preserving FC (average of 5)
    from encephagen.experiments.experiments_common import _degree_preserving_rewire
    null_fcs = []
    for i in range(5):
        rc = _degree_preserving_rewire(connectome, seed=400 + i)
        sim = BrainSimulator(rc, global_coupling=G, params=params)
        rr = sim.simulate(duration=5000, dt=0.1, transient=1000, seed=42)
        null_fcs.append(np.corrcoef(rr.E.T))
    null_fc = np.mean(null_fcs, axis=0)

    diff_fc = real_fc - null_fc

    # Group-averaged FC
    group_order = ["basal_ganglia", "prefrontal", "thalamus", "hippocampus", "sensory", "motor", "other"]
    group_labels_short = ["BG", "PFC", "Thal", "Hipp", "Sens", "Mot", "Other"]

    def _group_fc(fc_matrix):
        n = len(group_order)
        gfc = np.zeros((n, n))
        for i, g1 in enumerate(group_order):
            for j, g2 in enumerate(group_order):
                idx1, idx2 = groups.get(g1, []), groups.get(g2, [])
                if idx1 and idx2:
                    vals = [fc_matrix[a, b] for a in idx1 for b in idx2 if a != b]
                    gfc[i, j] = np.mean(vals) if vals else 0
        return gfc

    real_gfc = _group_fc(real_fc)
    null_gfc = _group_fc(null_fc)
    diff_gfc = real_gfc - null_gfc

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    vmax = max(abs(real_gfc).max(), abs(null_gfc).max())

    for ax, data, title in [
        (axes[0], real_gfc, "Real Connectome"),
        (axes[1], null_gfc, "Degree-Preserving\nRandom Wiring"),
        (axes[2], diff_gfc, "Difference\n(Real − Random)"),
    ]:
        if "Difference" in title:
            cmap = "RdBu_r"
            vm = max(abs(diff_gfc).max(), 0.005)
            im = ax.imshow(data, cmap=cmap, vmin=-vm, vmax=vm, aspect="auto")
        else:
            cmap = "coolwarm"
            im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(group_labels_short)))
        ax.set_xticklabels(group_labels_short, fontsize=10)
        ax.set_yticks(range(len(group_labels_short)))
        ax.set_yticklabels(group_labels_short, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

        # Annotate values
        for i in range(len(group_order)):
            for j in range(len(group_order)):
                val = data[i, j]
                color = "white" if abs(val) > vmax * 0.6 else "black"
                if "Difference" in title:
                    color = "white" if abs(val) > vm * 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

        # Grid
        for i in range(len(group_order) + 1):
            ax.axhline(i - 0.5, color="white", linewidth=1.5)
            ax.axvline(i - 0.5, color="white", linewidth=1.5)

        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    fig.text(0.5, 0.98,
             "What Specific Wiring Contributes: Functional Connectivity Patterns",
             ha="center", fontsize=15, fontweight="bold", color="#1a1a2e")
    fig.text(0.5, 0.935,
             "Group-averaged FC at G=0.01  |  Difference shows what degree distribution alone cannot explain",
             ha="center", fontsize=10, color="#777777")

    plt.subplots_adjust(wspace=0.35, top=0.87, bottom=0.08)

    fig.savefig("figures/fig3_fc_comparison.png")
    fig.savefig("figures/fig3_fc_comparison.svg")
    plt.close()
    print("  Saved: figures/fig3_fc_comparison.png")


if __name__ == "__main__":
    figure1_phase_transition()
    figure2_hierarchy_bar()

    # Figure 3 needs the rewiring function — try import, skip if not available
    try:
        figure3_fc_comparison()
    except Exception as e:
        print(f"  Figure 3 skipped: {e}")
        print("  Running standalone FC comparison instead...")

        # Inline version without external import
        connectome = Connectome.from_bundled("tvb96")
        groups = _classify_tvb76_regions(connectome.labels)
        params = _params()
        G = 0.01

        sim = BrainSimulator(connectome, global_coupling=G, params=params)
        r = sim.simulate(duration=5000, dt=0.1, transient=1000, seed=42)
        real_fc = np.corrcoef(r.E.T)

        # Simple random rewiring
        rng = np.random.default_rng(42)
        w = connectome.weights.copy()
        n = w.shape[0]
        rows, cols = np.where(w > 0)
        edges = list(zip(rows.tolist(), cols.tolist()))
        wts = [float(w[rr, c]) for rr, c in edges]
        ne = len(edges)
        for _ in range(10 * ne):
            i1, i2 = rng.choice(ne, size=2, replace=False)
            a, b = edges[i1]
            c, d = edges[i2]
            if a == d or c == b or w[a, d] > 0 or w[c, b] > 0:
                continue
            w[a, b], w[c, d] = 0, 0
            w[a, d], w[c, b] = wts[i1], wts[i2]
            edges[i1], edges[i2] = (a, d), (c, b)
        null_conn = Connectome.from_numpy(w, list(connectome.labels))

        sim2 = BrainSimulator(null_conn, global_coupling=G, params=params)
        r2 = sim2.simulate(duration=5000, dt=0.1, transient=1000, seed=42)
        null_fc = np.corrcoef(r2.E.T)

        diff_fc = real_fc - null_fc

        group_order = ["basal_ganglia", "prefrontal", "thalamus", "hippocampus", "sensory", "motor", "other"]
        group_labels_short = ["BG", "PFC", "Thal", "Hipp", "Sens", "Mot", "Other"]

        def _gfc(fc):
            ng = len(group_order)
            out = np.zeros((ng, ng))
            for i, g1 in enumerate(group_order):
                for j, g2 in enumerate(group_order):
                    i1, i2 = groups.get(g1, []), groups.get(g2, [])
                    if i1 and i2:
                        vals = [fc[a, b] for a in i1 for b in i2 if a != b]
                        out[i, j] = np.mean(vals) if vals else 0
            return out

        real_gfc = _gfc(real_fc)
        null_gfc = _gfc(null_fc)
        diff_gfc = real_gfc - null_gfc

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        vmax = max(abs(real_gfc).max(), abs(null_gfc).max())

        for ax, data, title in [
            (axes[0], real_gfc, "Real Connectome"),
            (axes[1], null_gfc, "Degree-Preserving\nRandom"),
            (axes[2], diff_gfc, "Difference\n(Real − Random)"),
        ]:
            if "Difference" in title:
                vm = max(abs(diff_gfc).max(), 0.005)
                im = ax.imshow(data, cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
            else:
                im = ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")

            ax.set_xticks(range(len(group_labels_short)))
            ax.set_xticklabels(group_labels_short, fontsize=10)
            ax.set_yticks(range(len(group_labels_short)))
            ax.set_yticklabels(group_labels_short, fontsize=10)
            ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

            for i in range(len(group_order)):
                for j in range(len(group_order)):
                    val = data[i, j]
                    threshold = vm * 0.6 if "Difference" in title else vmax * 0.6
                    color = "white" if abs(val) > threshold else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=8, color=color, fontweight="bold")

            for i in range(len(group_order) + 1):
                ax.axhline(i - 0.5, color="white", linewidth=1.5)
                ax.axvline(i - 0.5, color="white", linewidth=1.5)

            fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

        fig.text(0.5, 0.98,
                 "What Specific Wiring Contributes: Functional Connectivity Patterns",
                 ha="center", fontsize=15, fontweight="bold", color="#1a1a2e")
        fig.text(0.5, 0.935,
                 "Group-averaged FC at G=0.01  |  Difference = what degree distribution alone cannot explain",
                 ha="center", fontsize=10, color="#777777")
        plt.subplots_adjust(wspace=0.35, top=0.87, bottom=0.08)

        fig.savefig("figures/fig3_fc_comparison.png")
        fig.savefig("figures/fig3_fc_comparison.svg")
        plt.close()
        print("  Saved: figures/fig3_fc_comparison.png")

    print("\nAll figures generated!")
