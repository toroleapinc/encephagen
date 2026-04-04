"""Statistical utilities for multi-comparison correction.

All experiments with multiple hypothesis tests MUST use FDR correction.
"""

import numpy as np


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR correction.

    Args:
        p_values: list of raw p-values from multiple tests
        alpha: desired FDR level (default 0.05)

    Returns:
        list of booleans — True if test survives FDR correction
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort p-values and track original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    sorted_idx = [x[0] for x in indexed]
    sorted_p = [x[1] for x in indexed]

    # BH threshold: p_(k) <= (k/n) * alpha
    significant = [False] * n
    max_k = -1
    for k in range(n):
        threshold = (k + 1) / n * alpha
        if sorted_p[k] <= threshold:
            max_k = k

    # All tests up to max_k are significant
    if max_k >= 0:
        for k in range(max_k + 1):
            significant[sorted_idx[k]] = True

    return significant


def report_with_fdr(labels: list[str], p_values: list[float],
                    alpha: float = 0.05) -> str:
    """Format a multi-comparison report with FDR correction."""
    survives = benjamini_hochberg(p_values, alpha)
    lines = []
    lines.append(f"  FDR correction (Benjamini-Hochberg, alpha={alpha}):")
    for i, (label, p, surv) in enumerate(zip(labels, p_values, survives)):
        status = "SURVIVES FDR" if surv else "does NOT survive FDR"
        lines.append(f"    {label:<40} p={p:.4f}  {status}")
    n_surv = sum(survives)
    lines.append(f"  {n_surv}/{len(p_values)} tests survive FDR correction")
    return "\n".join(lines)
