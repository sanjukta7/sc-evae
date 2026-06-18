"""
Train-time proxy perturbation metrics.

These are the fast, noisy, per-batch single-cell signals logged during training
for monitoring only (the training loop prefixes them ``proxy_*``). They are NOT
the paper's reported numbers — those come from the pseudobulk pipeline in
``scripts/eval.py`` (cell-eval metrics in ``state_metrics.py`` + distribution
metrics in ``distribution.py``).

The training loop (``train_utils.run_validation``) tracks two DEG-focused axes:
delta correlation (``batch_pearson_delta_deg20``) and magnitude / R²
(``batch_r2_deg20``), both restricted to each perturbation's top-20 DEGs via a
precomputed DEG table (``deg_table.py``).
"""

from __future__ import annotations

import math

from torch import Tensor


def _pearson(x: Tensor, y: Tensor) -> float:
    """Pearson correlation between two 1-D float tensors."""
    if x.numel() < 2:
        return float("nan")
    xc = x.float() - x.float().mean()
    yc = y.float() - y.float().mean()
    num = (xc * yc).sum()
    denom = (xc.pow(2).sum() * yc.pow(2).sum()).sqrt().clamp_min(1e-8)
    return (num / denom).item()


def pearson_delta_deg20(
    pred: Tensor, ctrl: Tensor, true_delta: Tensor, mask: Tensor
) -> float:
    """Pearson r between predicted delta and true delta on the top-k DEG positions.

    Correlates ``(pred − ctrl)`` vs ``(true − ctrl)`` restricted to the top-k
    abs-Δ DEG mask. Removing the ctrl baseline means the value isn't anchored by
    housekeeping expression — a predict-ctrl model scores ~0.
    """
    pd = (pred.float() - ctrl.float())[mask]
    td = true_delta.float()[mask]
    return _pearson(pd, td)


def r2_deg20(pred: Tensor, true: Tensor, mask: Tensor) -> float:
    """
    Coefficient of determination (R²) on the top-k DEG positions.

    R² = 1 - SS_res / SS_tot.  Can be negative when the model is worse than
    predicting the mean.

    Parameters
    ----------
    pred, true : (G,) float32 — mean expression for one perturbation.
    mask       : (G,) bool — True at the top-k DEG positions.

    Returns
    -------
    float — R² (can be negative), or NaN if SS_tot < 1e-8 (constant true).
    """
    x = pred[mask].float()
    y = true[mask].float()
    if x.numel() < 2:
        return float("nan")
    ss_res = (x - y).pow(2).sum()
    ss_tot = (y - y.mean()).pow(2).sum()
    if ss_tot.item() < 1e-8:
        return float("nan")
    return (1.0 - ss_res / ss_tot).item()


# ---------------------------------------------------------------------------
# Batch proxy metrics  (inputs: (B, G) tensors — for use during training)
# ---------------------------------------------------------------------------


def batch_r2_deg20(
    pred: Tensor,
    true: Tensor,
    pert_keys: list,
    deg_table: dict,
) -> float:
    """
    Mean R²-DEG-20 over a batch, using a precomputed DEG table.

    Parameters
    ----------
    pred      : (B, G) float32.
    true      : (B, G) float32.
    pert_keys : list of length B — keys into *deg_table*.
    deg_table : dict mapping pert_key → {"mask": (G,) bool Tensor, ...}.

    Returns
    -------
    float — mean R²-DEG-20, or NaN if no valid samples.
    """
    scores = []
    for i, key in enumerate(pert_keys):
        if key not in deg_table:
            continue
        mask = deg_table[key]["mask"]
        s = r2_deg20(pred[i], true[i], mask)
        if not math.isnan(s):
            scores.append(s)
    return float(sum(scores) / len(scores)) if scores else float("nan")


def batch_pearson_delta_deg20(
    pred: Tensor,
    true: Tensor,
    pert_keys: list,
    deg_table: dict,
) -> float:
    """
    Mean Pearson-Δ-DEG-20 over a batch, using the precomputed DEG table.

    Correlates **deltas** — ``(pred − ctrl)`` vs ``(true − ctrl)`` — restricted
    to each pert's top-20 abs-Δ DEG mask, with ``ctrl`` from the table's
    ``ctrl_mean``. Perturbations not in the table are skipped.
    """
    scores = []
    for i, key in enumerate(pert_keys):
        if key not in deg_table:
            continue
        ctrl = deg_table[key]["ctrl_mean"]
        mask = deg_table[key]["mask"]
        true_delta = true[i] - ctrl
        s = pearson_delta_deg20(pred[i], ctrl, true_delta, mask)
        if not math.isnan(s):
            scores.append(s)
    return float(sum(scores) / len(scores)) if scores else float("nan")
