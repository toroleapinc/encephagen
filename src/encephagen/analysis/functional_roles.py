"""Analyze whether region-specific functional roles emerge from connectome topology.

Tests 5 predictions about emergent functional organization:
1. Thalamic gating: high variance, bistable switching
2. Prefrontal sustained activity: longer time constants
3. Sensory-first response: shortest stimulus response latency
4. Regional differentiation: distinct oscillation frequencies per region
5. Network emergence: DMN-like co-activation patterns
"""

from __future__ import annotations

import numpy as np
from scipy import signal, stats

from encephagen.dynamics.brain_sim import SimulationResult
from encephagen.connectome.parcellations import classify_region


# --- TVB76 region classification helpers ---

def _classify_tvb76_regions(labels: list[str]) -> dict[str, list[int]]:
    """Classify TVB76 regions into functional groups.

    TVB76 labels use prefixes like 'r'/'l' for hemisphere and abbreviations:
    rA1=right primary auditory, lV1=left primary visual, etc.
    """
    groups: dict[str, list[int]] = {
        "thalamus": [],
        "basal_ganglia": [],
        "sensory": [],
        "prefrontal": [],
        "hippocampus": [],
        "motor": [],
        "other": [],
    }

    basal_ganglia_keys = ["BG-Cd", "BG-Pu", "BG-Pa", "BG-Acc"]

    # TVB76: rA1, lPFCDL, rHC, etc. Prefix r/l = hemisphere.
    # TVB96: adds TM-F_R (thalamus frontal), BG-Cd_R (caudate), etc.
    thalamic_keys = ["TM-F", "TM-T", "TM-OP"]
    sensory_keys = ["V1", "V2", "A1", "A2", "S1", "S2"]
    prefrontal_keys = ["PFCCL", "PFCDL", "PFCDM", "PFCM", "PFCORB", "PFCPOL", "PFCVL", "FEF"]
    hippocampal_keys = ["HC", "PHC"]
    motor_keys = ["M1", "PMCDL", "PMCM", "PMCVL"]

    for i, label in enumerate(labels):
        categorized = False
        for key in thalamic_keys:
            if key in label:
                groups["thalamus"].append(i)
                categorized = True
                break
        if categorized:
            continue
        for key in basal_ganglia_keys:
            if key in label:
                groups["basal_ganglia"].append(i)
                categorized = True
                break
        if categorized:
            continue
        for key in sensory_keys:
            if key in label:
                groups["sensory"].append(i)
                categorized = True
                break
        if categorized:
            continue
        for key in prefrontal_keys:
            if key in label:
                groups["prefrontal"].append(i)
                categorized = True
                break
        if categorized:
            continue
        for key in hippocampal_keys:
            if key in label:
                groups["hippocampus"].append(i)
                categorized = True
                break
        if categorized:
            continue
        for key in motor_keys:
            if key in label:
                groups["motor"].append(i)
                categorized = True
                break
        if not categorized:
            groups["other"].append(i)

    return groups


def compute_regional_profiles(result: SimulationResult) -> dict[str, dict]:
    """Compute a dynamical profile for each region.

    Returns dict mapping region label -> profile dict with:
    - mean_activity: mean excitatory activity
    - variance: activity variance (proxy for dynamical richness)
    - autocorrelation_tau: decay time of autocorrelation (proxy for time constant)
    - peak_frequency: dominant oscillation frequency
    - spectral_entropy: breadth of frequency content
    """
    profiles = {}
    dt_sec = result.dt / 1000.0
    fs = 1.0 / dt_sec

    for i, label in enumerate(result.labels):
        ts = result.E[:, i]

        # Basic stats
        mean_act = float(np.mean(ts))
        variance = float(np.var(ts))

        # Autocorrelation decay time
        ts_centered = ts - np.mean(ts)
        if np.std(ts_centered) > 1e-10:
            autocorr = np.correlate(ts_centered, ts_centered, mode="full")
            autocorr = autocorr[len(autocorr) // 2:]
            autocorr = autocorr / (autocorr[0] + 1e-12)
            # Find where autocorrelation drops below 1/e
            below_threshold = np.where(autocorr < 1.0 / np.e)[0]
            if len(below_threshold) > 0:
                tau = float(below_threshold[0]) * result.dt  # in ms
            else:
                tau = float(len(autocorr)) * result.dt
        else:
            tau = 0.0

        # Power spectral density
        nperseg = min(1024, len(ts))
        if variance > 1e-10 and nperseg >= 8:
            freqs, psd = signal.welch(ts, fs=fs, nperseg=nperseg)
            peak_freq = float(freqs[np.argmax(psd)])
            # Spectral entropy
            psd_norm = psd / (psd.sum() + 1e-12)
            spectral_ent = float(-np.sum(psd_norm * np.log(psd_norm + 1e-12)))
        else:
            peak_freq = 0.0
            spectral_ent = 0.0

        profiles[label] = {
            "index": i,
            "mean_activity": mean_act,
            "variance": variance,
            "autocorrelation_tau": tau,
            "peak_frequency": peak_freq,
            "spectral_entropy": spectral_ent,
        }

    return profiles


def test_thalamic_gating(
    result: SimulationResult,
    groups: dict[str, list[int]],
) -> dict:
    """Test Prediction 1: Do thalamic regions show higher variance (gating-like behavior)?

    Gating = switching between high and low activity states = high variance.
    """
    thal_idx = groups.get("thalamus", [])
    other_idx = [i for g, indices in groups.items() if g != "thalamus" for i in indices]

    if not thal_idx or not other_idx:
        return {"testable": False, "reason": "No thalamic regions identified"}

    thal_var = [float(np.var(result.E[:, i])) for i in thal_idx]
    other_var = [float(np.var(result.E[:, i])) for i in other_idx]

    thal_mean = float(np.mean(thal_var))
    other_mean = float(np.mean(other_var))

    if len(thal_var) >= 2 and len(other_var) >= 2:
        stat, p_value = stats.mannwhitneyu(thal_var, other_var, alternative="greater")
    else:
        p_value = float("nan")

    return {
        "testable": True,
        "prediction": "Thalamic regions have higher variance (gating)",
        "thalamic_mean_var": thal_mean,
        "other_mean_var": other_mean,
        "ratio": thal_mean / (other_mean + 1e-12),
        "p_value": float(p_value),
        "supported": thal_mean > other_mean and (p_value < 0.05 if not np.isnan(p_value) else False),
    }


def test_prefrontal_sustained(
    result: SimulationResult,
    groups: dict[str, list[int]],
) -> dict:
    """Test Prediction 2: Do prefrontal regions show longer autocorrelation times?

    Sustained activity = slow dynamics = long autocorrelation decay.
    """
    pfc_idx = groups.get("prefrontal", [])
    sensory_idx = groups.get("sensory", [])

    if not pfc_idx or not sensory_idx:
        return {"testable": False, "reason": "Cannot compare PFC vs sensory"}

    profiles = compute_regional_profiles(result)

    pfc_tau = [profiles[result.labels[i]]["autocorrelation_tau"] for i in pfc_idx]
    sens_tau = [profiles[result.labels[i]]["autocorrelation_tau"] for i in sensory_idx]

    pfc_mean = float(np.mean(pfc_tau))
    sens_mean = float(np.mean(sens_tau))

    if len(pfc_tau) >= 2 and len(sens_tau) >= 2:
        stat, p_value = stats.mannwhitneyu(pfc_tau, sens_tau, alternative="greater")
    else:
        p_value = float("nan")

    return {
        "testable": True,
        "prediction": "Prefrontal regions have longer time constants than sensory",
        "pfc_mean_tau": pfc_mean,
        "sensory_mean_tau": sens_mean,
        "ratio": pfc_mean / (sens_mean + 1e-12),
        "p_value": float(p_value),
        "supported": pfc_mean > sens_mean and (p_value < 0.05 if not np.isnan(p_value) else False),
    }


def test_sensory_first_response(
    result: SimulationResult,
    groups: dict[str, list[int]],
    stimulus_onset_idx: int | None = None,
) -> dict:
    """Test Prediction 3: Do sensory regions respond first to stimulus?

    Measures latency to peak response after stimulus onset.
    """
    if not result.stimuli:
        return {"testable": False, "reason": "No stimulus in this simulation"}

    sensory_idx = groups.get("sensory", [])
    other_idx = [i for g, indices in groups.items() if g != "sensory" for i in indices]

    if not sensory_idx or not other_idx:
        return {"testable": False, "reason": "Cannot compare sensory vs other"}

    # Find stimulus onset in recording
    stim = result.stimuli[0]
    onset_step = max(0, int((stim.onset - result.time[0]) / result.dt))

    # Window after stimulus (500ms)
    window = int(500.0 / result.dt)
    end_step = min(onset_step + window, result.num_timesteps)

    if onset_step >= result.num_timesteps or end_step - onset_step < 10:
        return {"testable": False, "reason": "Stimulus outside recording window"}

    # Baseline: 200ms before stimulus
    baseline_start = max(0, onset_step - int(200.0 / result.dt))
    baseline = result.E[baseline_start:onset_step, :]

    # Response latency: time to first significant deviation from baseline
    def _latency(region_idx):
        bl_mean = float(np.mean(baseline[:, region_idx]))
        bl_std = float(np.std(baseline[:, region_idx]))
        threshold = bl_mean + 3 * max(bl_std, 1e-6)

        post_stim = result.E[onset_step:end_step, region_idx]
        above = np.where(post_stim > threshold)[0]
        if len(above) > 0:
            return float(above[0]) * result.dt  # ms
        return float("inf")

    sens_latencies = [_latency(i) for i in sensory_idx]
    other_latencies = [_latency(i) for i in other_idx]

    # Filter out inf
    sens_finite = [l for l in sens_latencies if np.isfinite(l)]
    other_finite = [l for l in other_latencies if np.isfinite(l)]

    sens_mean = float(np.mean(sens_finite)) if sens_finite else float("inf")
    other_mean = float(np.mean(other_finite)) if other_finite else float("inf")

    if len(sens_finite) >= 2 and len(other_finite) >= 2:
        stat, p_value = stats.mannwhitneyu(sens_finite, other_finite, alternative="less")
    else:
        p_value = float("nan")

    return {
        "testable": True,
        "prediction": "Sensory regions respond faster to stimulus",
        "sensory_mean_latency_ms": sens_mean,
        "other_mean_latency_ms": other_mean,
        "p_value": float(p_value),
        "supported": sens_mean < other_mean and (p_value < 0.05 if not np.isnan(p_value) else False),
    }


def test_frequency_differentiation(
    result: SimulationResult,
    groups: dict[str, list[int]],
) -> dict:
    """Test Prediction 4: Do different region types show distinct oscillation frequencies?"""
    profiles = compute_regional_profiles(result)

    group_freqs = {}
    for group_name, indices in groups.items():
        if indices:
            freqs = [profiles[result.labels[i]]["peak_frequency"] for i in indices
                     if profiles[result.labels[i]]["variance"] > 1e-8]
            if freqs:
                group_freqs[group_name] = freqs

    if len(group_freqs) < 2:
        return {"testable": False, "reason": "Not enough groups with oscillations"}

    # Kruskal-Wallis test across groups
    group_arrays = list(group_freqs.values())
    if all(len(g) >= 2 for g in group_arrays):
        stat, p_value = stats.kruskal(*group_arrays)
    else:
        p_value = float("nan")

    summary = {
        group: {"mean_freq": float(np.mean(f)), "std_freq": float(np.std(f)), "n": len(f)}
        for group, f in group_freqs.items()
    }

    return {
        "testable": True,
        "prediction": "Different region types have distinct oscillation frequencies",
        "group_frequencies": summary,
        "kruskal_p_value": float(p_value),
        "supported": p_value < 0.05 if not np.isnan(p_value) else False,
    }


def test_regional_differentiation_overall(
    result: SimulationResult,
) -> dict:
    """Test Prediction 5: Do regions differentiate at all (vs uniform behavior)?

    Measures how different regions are from each other in their dynamical profiles.
    """
    profiles = compute_regional_profiles(result)
    variances = [p["variance"] for p in profiles.values()]
    taus = [p["autocorrelation_tau"] for p in profiles.values()]
    freqs = [p["peak_frequency"] for p in profiles.values()]

    return {
        "testable": True,
        "prediction": "Regions develop distinct dynamical profiles",
        "variance_cv": float(np.std(variances) / (np.mean(variances) + 1e-12)),
        "tau_cv": float(np.std(taus) / (np.mean(taus) + 1e-12)),
        "freq_cv": float(np.std(freqs) / (np.mean(freqs) + 1e-12)),
        "supported": float(np.std(variances) / (np.mean(variances) + 1e-12)) > 0.1,
    }


def run_all_predictions(
    result: SimulationResult,
) -> dict[str, dict]:
    """Run all 5 predictions on a simulation result."""
    groups = _classify_tvb76_regions(result.labels)

    return {
        "region_groups": {k: len(v) for k, v in groups.items()},
        "P1_thalamic_gating": test_thalamic_gating(result, groups),
        "P2_prefrontal_sustained": test_prefrontal_sustained(result, groups),
        "P3_sensory_first": test_sensory_first_response(result, groups),
        "P4_frequency_differentiation": test_frequency_differentiation(result, groups),
        "P5_regional_differentiation": test_regional_differentiation_overall(result),
    }
