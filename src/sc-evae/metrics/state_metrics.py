"""State-faithful metrics — match the cell-eval implementations used in
Adduri et al. 2025 (Fig. 2 panels D, F, H, I, J).

Each function consumes either a per-pert long-form DE table (output of
``build_de_table``) or per-pert pseudobulks (for the discrimination score),
and returns either ``dict[pert] -> float`` or a single scalar — matching
cell-eval's API shape.

References
----------
- ``cell_eval.metrics._anndata.discrimination_score`` (Wu et al. 2024 inverse-rank)
- ``cell_eval.metrics._de.de_overlap_metric`` (overlap_at_n)
- ``cell_eval.metrics._de.DESpearmanLFC`` (de_spearman_lfc_sig)
- ``cell_eval.metrics._de.DESpearmanSignificant`` (de_spearman_sig)
- ``cell_eval.metrics._de.compute_pr_auc`` (pr_auc)
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import average_precision_score, pairwise_distances

# ---------------------------------------------------------------------------
# Fig 2D — Perturbation discrimination score (Wu et al. 2024 normalized rank)
# ---------------------------------------------------------------------------


def discrimination_score_l1(
    real_pseudobulks: np.ndarray,
    pred_pseudobulks: np.ndarray,
    ctrl_pseudobulk: np.ndarray,
    perts: Sequence[str],
    metric: str = "l1",
    gene_names: Sequence[str] | None = None,
    exclude_target_gene: bool = False,
) -> dict[str, float]:
    """Per-pert normalized-rank discrimination score (cell-eval convention).

    For each perturbation p:
      1. Compute predicted effect ``pred_delta_p = pred_pseudobulks[p] - ctrl``.
      2. Compute observed effects ``real_delta_q = real_pseudobulks[q] - ctrl``
         for ALL perturbations q.
      3. Score q's = pairwise distance(pred_delta_p, real_delta_q).
      4. Rank q's ascending by distance; find rank of p; return ``1 - rank/n_perts``.

    Best score = 1.0; random ≈ 0.5; worst = 0.0.

    Parameters
    ----------
    real_pseudobulks, pred_pseudobulks:
        ``(n_perts, n_genes)`` pseudobulks — same row order as ``perts``.
        cell-eval / STATE compute this metric on the **log1p** space (the
        arithmetic mean of the log-normed .X, via ``group_by().mean()`` — no
        expm1), so feed log1p per-pert means here, NOT expm1'd count-space
        pseudobulk, to match their reported numbers.
    ctrl_pseudobulk:
        ``(n_genes,)`` control pseudobulk in the same (log1p) space.
    perts:
        Length ``n_perts`` perturbation labels (must align with row order).
    metric:
        Forwarded to ``sklearn.metrics.pairwise_distances`` (``"l1"`` is the
        State paper convention; ``"l2"``, ``"cosine"`` also valid).
    gene_names, exclude_target_gene:
        When ``exclude_target_gene=True``, drops the column whose ``gene_name``
        matches the perturbation label before scoring (Replogle convention —
        guide knocks down its own target). Set False for cytokine / chemical
        perturbations where the perturbation is not a gene.

        NOTE: the default here is ``False``, but the cell-eval / State (Fig 2D)
        convention is ``exclude_target_gene=True`` — without it, a CRISPR
        knockdown's own large self-change dominates the L1 distance and inflates
        the score. ``scripts/eval.py`` always passes ``True`` (with aligned
        ``gene_names``); pass ``True`` too if you call this directly on
        gene-targeting (e.g. Replogle) data. For cytokine names that don't match
        any gene symbol, ``True`` is a harmless no-op.
    """
    real_pseudobulks = np.asarray(real_pseudobulks, dtype=np.float64)
    pred_pseudobulks = np.asarray(pred_pseudobulks, dtype=np.float64)
    ctrl_pseudobulk = np.asarray(ctrl_pseudobulk, dtype=np.float64)
    perts = np.asarray(perts)
    n_perts, n_genes = real_pseudobulks.shape
    if pred_pseudobulks.shape != (n_perts, n_genes):
        raise ValueError(
            f"pred_pseudobulks shape {pred_pseudobulks.shape} != "
            f"real_pseudobulks shape {real_pseudobulks.shape}."
        )
    if ctrl_pseudobulk.shape != (n_genes,):
        raise ValueError(
            f"ctrl_pseudobulk shape {ctrl_pseudobulk.shape} != ({n_genes},)."
        )

    real_delta = real_pseudobulks - ctrl_pseudobulk[None, :]
    pred_delta = pred_pseudobulks - ctrl_pseudobulk[None, :]

    if exclude_target_gene:
        if gene_names is None:
            raise ValueError("exclude_target_gene=True requires gene_names.")
        gene_names = np.asarray(gene_names)

    out: dict[str, float] = {}
    for p_idx, p in enumerate(perts):
        if exclude_target_gene:
            mask = gene_names != p
            r = real_delta[:, mask]
            d = pred_delta[p_idx, mask].reshape(1, -1)
        else:
            r = real_delta
            d = pred_delta[p_idx].reshape(1, -1)

        distances = pairwise_distances(r, d, metric=metric).flatten()  # (n_perts,)
        sorted_idx = np.argsort(distances)
        rank = int(np.flatnonzero(sorted_idx == p_idx)[0])
        norm_rank = rank / n_perts
        out[str(p)] = 1.0 - norm_rank
    return out


# ---------------------------------------------------------------------------
# Fig 2I — Overlap@N on FDR-significant DE genes ranked by abs(log2FC)
# ---------------------------------------------------------------------------


def overlap_at_n(
    real_de: pd.DataFrame,
    pred_de: pd.DataFrame,
    n: int | None = None,
    fdr_threshold: float = 0.05,
) -> dict[str, float]:
    """Per-pert overlap of top-N FDR-significant genes by ``|log2_fold_change|``.

    For each pert: take genes with ``fdr < fdr_threshold`` (in real and pred
    independently), sort each by ``|log2_fold_change|`` descending, take the
    top-N from each, return ``|intersection| / N``.

    When ``n=None`` (cell-eval default for variable-N evaluation), N is set
    per-pert to the number of real-significant genes for that pert — matching
    State Fig 2I where k = #true_DEGs.

    Returns ``{pert: overlap}`` (NaN when N=0 for that pert).
    """
    real_grp = real_de[real_de["fdr"] < fdr_threshold].groupby("target")
    pred_grp = pred_de[pred_de["fdr"] < fdr_threshold].groupby("target")

    out: dict[str, float] = {}
    for target, real_part in real_grp:
        if n is None:
            k = int(real_part.shape[0])
        else:
            k = int(n)
        if k == 0:
            out[str(target)] = float("nan")
            continue

        real_top = (
            real_part.assign(_abs=lambda df: df["log2_fold_change"].abs())
            .sort_values("_abs", ascending=False)
            .head(k)["gene"]
            .to_numpy()
        )
        pred_part = pred_grp.get_group(target) if target in pred_grp.groups else None
        if pred_part is None or pred_part.shape[0] == 0:
            out[str(target)] = 0.0
            continue
        pred_top = (
            pred_part.assign(_abs=lambda df: df["log2_fold_change"].abs())
            .sort_values("_abs", ascending=False)
            .head(k)["gene"]
            .to_numpy()
        )
        out[str(target)] = float(np.intersect1d(real_top, pred_top).size) / k
    return out


# ---------------------------------------------------------------------------
# Fig 2H — Spearman of log2FC over real-FDR-significant gene set
# ---------------------------------------------------------------------------


def de_spearman_lfc_sig(
    real_de: pd.DataFrame,
    pred_de: pd.DataFrame,
    fdr_threshold: float = 0.05,
) -> dict[str, float]:
    """Per-pert Spearman of ``log2_fold_change`` over the real-FDR-significant set.

    For each pert:
      1. Take ``real`` rows where ``fdr < fdr_threshold``.
      2. Left-join with ``pred`` on (target, gene); fill missing pred logFC with 0.
      3. Drop pairs where either log2FC is non-finite (±inf / NaN — arises from
         zero-count genes at ``de_epsilon=0``; a rank correlation is undefined
         on inf, and a single such gene would otherwise NaN-collapse the whole
         pert under ``scipy.spearmanr``'s default ``nan_policy='propagate'``).
      4. Spearman correlation between ``real.log2FC`` and ``pred.log2FC``
         on the remaining finite pairs.

    Returns ``{pert: spearman}`` (NaN when fewer than 2 finite sig genes).
    """
    real_sig = real_de[real_de["fdr"] < fdr_threshold]
    out: dict[str, float] = {}
    pred_indexed = pred_de.set_index(["target", "gene"])["log2_fold_change"]
    for target, part in real_sig.groupby("target"):
        x = part["log2_fold_change"].to_numpy(dtype=np.float64)
        y = np.array(
            [pred_indexed.get((target, g), 0.0) for g in part["gene"].to_numpy()],
            dtype=np.float64,
        )
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
            out[str(target)] = float("nan")
            continue
        out[str(target)] = float(scipy.stats.spearmanr(x, y).correlation)
    return out


# ---------------------------------------------------------------------------
# MAE of log2FC over real-FDR-significant set (DE-restricted; sc_evae-specific)
# ---------------------------------------------------------------------------
# Note: this is NOT the MAE column reported in scLDM Fig 13 / cell-eval `mae`,
# which is a pseudobulk-level mean_absolute_error across all genes in
# expression space (see ``mae_pseudobulk`` below). This metric is a sc_evae
# addition that restricts MAE to the real-FDR-significant gene set in log2FC
# space.


def mae_log2fc_sig(
    real_de: pd.DataFrame,
    pred_de: pd.DataFrame,
    fdr_threshold: float = 0.05,
) -> dict[str, float]:
    """Per-pert mean absolute error of ``log2_fold_change`` restricted to the
    real-FDR-significant gene set.

    For each pert:
      1. Take ``real`` rows where ``fdr < fdr_threshold``.
      2. Left-join with ``pred`` on (target, gene); fill missing pred logFC
         with 0.
      3. Drop pairs where either log2FC is non-finite (±inf / NaN from
         zero-count genes at ``de_epsilon=0``; one such gene would otherwise
         blow the MAE up to inf).
      4. ``MAE = mean(|real.log2FC − pred.log2FC|)`` over the remaining
         finite pairs.

    Returns ``{pert: mae}`` (NaN when no finite sig genes remain).
    """
    real_sig = real_de[real_de["fdr"] < fdr_threshold]
    pred_indexed = pred_de.set_index(["target", "gene"])["log2_fold_change"]
    out: dict[str, float] = {}
    for target, part in real_sig.groupby("target"):
        x = part["log2_fold_change"].to_numpy(dtype=np.float64)
        y = np.array(
            [pred_indexed.get((target, g), 0.0) for g in part["gene"].to_numpy()],
            dtype=np.float64,
        )
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if x.size == 0:
            out[str(target)] = float("nan")
            continue
        out[str(target)] = float(np.mean(np.abs(x - y)))
    return out


# ---------------------------------------------------------------------------
# Fig 2J — Across-pert Spearman of #significant-genes-per-pert
# ---------------------------------------------------------------------------


def de_spearman_sig(
    real_de: pd.DataFrame,
    pred_de: pd.DataFrame,
    fdr_threshold: float = 0.05,
) -> float:
    """Single scalar: Spearman of (#real-sig genes per pert) vs (#pred-sig genes per pert).

    **Left join on the real-significant perts** (cell-eval
    ``DESpearmanSignificant``: ``filt_real.join(filt_pred, how="left")
    .fill_null(0)``). Only perts with ≥1 real-FDR-sig gene are scored; their
    pred count is filled to 0 when pred has none. Perts with 0 real-sig but
    >0 pred-sig genes are **dropped** (a union/outer join would instead score
    them as ``(0, >0)`` and diverge from cell-eval's number). Returns ``1.0``
    when there are no real-sig perts at all (cell-eval's degenerate sentinel;
    unreachable on real data) and ``nan`` when fewer than 2 perts remain or
    either side has zero variance.
    """
    real_counts = (
        real_de[real_de["fdr"] < fdr_threshold]
        .groupby("target")
        .size()
        .rename("real_n")
    )
    pred_counts = (
        pred_de[pred_de["fdr"] < fdr_threshold]
        .groupby("target")
        .size()
        .rename("pred_n")
    )
    # Left join on real: index = real-sig perts only; reindex pred onto it and
    # fill missing pred counts with 0 (NOT an outer pd.concat, which would add
    # pred-only perts as (0, >0) and break parity with cell-eval).
    merged = real_counts.to_frame()
    merged["pred_n"] = pred_counts.reindex(merged.index).fillna(0)
    if merged.shape[0] == 0:
        return 1.0
    if merged.shape[0] < 2:
        return float("nan")
    x = merged["real_n"].to_numpy(dtype=np.float64)
    y = merged["pred_n"].to_numpy(dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(scipy.stats.spearmanr(x, y).correlation)


# ---------------------------------------------------------------------------
# Fig 2F — Precision-Recall AUC for FDR-significant gene recovery
# ---------------------------------------------------------------------------


def pr_auc(
    real_de: pd.DataFrame,
    pred_de: pd.DataFrame,
    fdr_threshold: float = 0.05,
    pred_fdr_clip: tuple[float, float] = (1e-10, 1.0),
) -> dict[str, float]:
    """Per-pert AP using real FDR-significance as labels and ``-log10(pred_fdr)``
    as scores (matches ``cell_eval.metrics._de.compute_pr_auc``).

    Returns ``{pert: ap}`` (NaN when no positives or no negatives in that pert).
    """
    real_indexed = real_de.set_index(["target", "gene"])["fdr"].rename("real_fdr")
    pred_indexed = pred_de.set_index(["target", "gene"])["fdr"].rename("pred_fdr")
    merged = pd.concat([real_indexed, pred_indexed], axis=1).reset_index()
    merged = merged.dropna(subset=["real_fdr"])
    merged["pred_fdr"] = (
        merged["pred_fdr"].fillna(1.0).clip(pred_fdr_clip[0], pred_fdr_clip[1])
    )
    merged["label"] = (merged["real_fdr"] < fdr_threshold).astype(np.int32)
    merged["nlp"] = -np.log10(merged["pred_fdr"].to_numpy())

    out: dict[str, float] = {}
    for target, part in merged.groupby("target"):
        y_true = part["label"].to_numpy()
        y_score = part["nlp"].to_numpy()
        if y_true.sum() == 0 or y_true.sum() == y_true.size:
            out[str(target)] = float("nan")
            continue
        out[str(target)] = float(average_precision_score(y_true, y_score))
    return out


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------


def aggregate_per_pert(
    scores: dict[str, float],
) -> dict[str, float]:
    """Mean ± std of a {pert: score} dict, ignoring NaNs."""
    valid = [v for v in scores.values() if not (v is None or math.isnan(v))]
    if not valid:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    arr = np.asarray(valid, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }
