"""Per-perturbation differential expression tables for State-faithful metrics.

We follow the conventions of ``ArcInstitute/pdex`` (the DE engine used by
``cell-eval``) so the resulting metrics line up with the State paper:

* Inputs are log1p(CP10K) cell × gene matrices (sc_evae's "metric space").
* Pseudobulk per gene = ``expm1(mean(log1p_cells))`` — geometric mean back-
  transformed to count space (``pdex.pseudobulk(geometric_mean=True,
  is_log1p=True)``).
* ``log2_fold_change`` per gene = ``log2((mean_pert + ε) / (mean_ctrl + ε))``
  with the same default ε=0 as pdex; pass ``epsilon=1e-9`` to dampen
  near-zero-denominator artifacts on sparse genes.
* P-values per gene from a two-sided Mann-Whitney U test on the original
  log1p values (``scipy.stats.mannwhitneyu`` with ``axis=0`` — vectorised
  across genes; rank-based so log1p vs raw doesn't matter).
* FDR per gene via Benjamini-Hochberg, applied **within each pert** across
  genes, using ``scipy.stats.false_discovery_control``.

The returned table is long-form (one row per (pert, gene) pair) so it fits
the cell-eval ``DEComparison`` shape that ``state_metrics`` consumes.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.stats
import torch

logger = logging.getLogger(__name__)


def _pseudobulk_count_space(cells_log1p: np.ndarray) -> np.ndarray:
    """Geometric pseudobulk back-transformed to count space: expm1(mean(log1p_cells)).

    Mirrors ``pdex.pseudobulk(geometric_mean=True, is_log1p=True)`` (default
    used by cell-eval). Inputs are log1p(CP10K) values; outputs are CP10K
    "count" values.
    """
    return np.expm1(cells_log1p.mean(axis=0))


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction (vector). Equivalent to
    ``scipy.stats.false_discovery_control(p, method='bh')`` but kept as a
    light-weight helper so the dependency path is obvious."""
    return scipy.stats.false_discovery_control(p, method="bh")


def compute_pert_de_table(
    pert_cells: np.ndarray | torch.Tensor,
    ctrl_cells: np.ndarray | torch.Tensor,
    gene_names: list[str] | np.ndarray,
    target: str,
    epsilon: float = 0.0,
    mwu_method: str = "auto",
) -> pd.DataFrame:
    """Build the per-(pert, gene) DE table for one perturbation.

    Returns a DataFrame with columns:
        target, gene, mean_pert, mean_ctrl, log2_fold_change, p_value, fdr.

    All means are in count space; log2_fold_change uses pdex's default
    ``epsilon=0.0``.
    """
    if isinstance(pert_cells, torch.Tensor):
        pert_cells = pert_cells.detach().cpu().numpy()
    if isinstance(ctrl_cells, torch.Tensor):
        ctrl_cells = ctrl_cells.detach().cpu().numpy()
    pert_cells = np.asarray(pert_cells, dtype=np.float64)
    ctrl_cells = np.asarray(ctrl_cells, dtype=np.float64)

    if pert_cells.ndim != 2 or ctrl_cells.ndim != 2:
        raise ValueError(
            f"compute_pert_de_table expects 2-D (n_cells, n_genes) inputs; got "
            f"pert={pert_cells.shape}, ctrl={ctrl_cells.shape}."
        )
    if pert_cells.shape[1] != ctrl_cells.shape[1]:
        raise ValueError(
            f"gene-axis mismatch: pert has {pert_cells.shape[1]} genes, "
            f"ctrl has {ctrl_cells.shape[1]}."
        )

    n_genes = pert_cells.shape[1]
    if n_genes != len(gene_names):
        raise ValueError(f"gene_names length {len(gene_names)} != n_genes {n_genes}.")

    mean_pert = _pseudobulk_count_space(pert_cells)
    mean_ctrl = _pseudobulk_count_space(ctrl_cells)

    # log2 fold change: log2((mean_pert + ε) / (mean_ctrl + ε)).
    # With ε=0, genes with mean_ctrl==0 produce inf — we leave that to the
    # downstream metric (cell-eval keeps the inf and lets it dominate
    # abs-log2-FC sorts; matches their default).
    with np.errstate(divide="ignore", invalid="ignore"):
        log2fc = np.log2((mean_pert + epsilon) / (mean_ctrl + epsilon))

    # Vectorised Mann-Whitney U over the gene axis. `auto` picks asymptotic
    # for our group sizes (hundreds–thousands of cells per group) which is
    # what pdex/numba_mwu uses too.
    mwu = scipy.stats.mannwhitneyu(
        pert_cells, ctrl_cells, axis=0, alternative="two-sided", method=mwu_method
    )
    p_values = np.asarray(mwu.pvalue)

    fdr = _bh_fdr(p_values)

    return pd.DataFrame(
        {
            "target": np.repeat(str(target), n_genes),
            "gene": np.asarray(gene_names),
            "mean_pert": mean_pert,
            "mean_ctrl": mean_ctrl,
            "log2_fold_change": log2fc,
            "p_value": p_values,
            "fdr": fdr,
        }
    )


def build_de_table(
    per_pert_cells: dict[str, np.ndarray | torch.Tensor],
    ctrl_cells: np.ndarray | torch.Tensor,
    gene_names: list[str] | np.ndarray,
    epsilon: float = 0.0,
    mwu_method: str = "auto",
    progress: bool = True,
) -> pd.DataFrame:
    """Build a long-form DE table across all perts vs the same ctrl baseline.

    Parameters
    ----------
    per_pert_cells:
        ``{pert_label: (N_p, G) cells in log1p(CP10K)}``.
    ctrl_cells:
        ``(N_c, G)`` control cells in log1p(CP10K). Used as the reference for
        every pert (matches cell-eval / pdex ``mode='ref'``).
    gene_names:
        Length-G gene-name array (must match the column order of every input).
    epsilon:
        Pseudocount added to both numerator and denominator before log2.
        Default 0.0 matches pdex; set to 1e-9 to dampen extreme fold changes
        on sparse genes.
    mwu_method:
        Forwarded to ``scipy.stats.mannwhitneyu``. ``"auto"`` picks asymptotic
        for the cell counts we typically have.
    progress:
        Show a tqdm progress bar over perts.
    """
    iterator: Iterable = sorted(per_pert_cells)
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(list(iterator), desc="building DE table")

    parts: list[pd.DataFrame] = []
    for label in iterator:
        cells = per_pert_cells[label]
        if isinstance(cells, list):  # caller passed list-of-tensors
            cells = torch.stack(
                [
                    c if isinstance(c, torch.Tensor) else torch.as_tensor(c)
                    for c in cells
                ]
            )
        if cells.shape[0] < 2:
            logger.warning(
                "DE: pert %r has %d cells (< 2) — skipping.", label, cells.shape[0]
            )
            continue
        parts.append(
            compute_pert_de_table(
                cells,
                ctrl_cells,
                gene_names,
                target=label,
                epsilon=epsilon,
                mwu_method=mwu_method,
            )
        )

    if not parts:
        return pd.DataFrame(
            columns=[
                "target",
                "gene",
                "mean_pert",
                "mean_ctrl",
                "log2_fold_change",
                "p_value",
                "fdr",
            ]
        )
    return pd.concat(parts, axis=0, ignore_index=True)
