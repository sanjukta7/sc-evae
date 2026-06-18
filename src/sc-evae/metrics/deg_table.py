"""
Build a table of top-k differentially expressed genes (DEGs) per perturbation.

The table is computed once from the training split and saved to disk.  During
training, it is loaded and passed to batch_pearson_delta_deg20 / batch_r2_deg20
for fast proxy metric computation on validation batches.

DEGs are ranked by mean absolute fold-change: |mean(pert) - mean(ctrl)|.
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def build_deg_table(
    counts: np.ndarray,
    perts: Union[list[str], np.ndarray],
    ctrl_label: str = "ctrl",
    top_k: int = 20,
) -> dict[str, dict]:
    """
    Build a per-perturbation DEG table from training-set expression data.

    DEGs are selected by the absolute mean difference in expression between
    the perturbed cells and the control cells.  This matches the GEARS
    convention (Roohani et al., Nature Biotechnology 2023).

    Parameters
    ----------
    counts     : (N, G) float32 numpy array — expression matrix (all cells).
    perts      : (N,) list or array of perturbation labels per cell.
    ctrl_label : str — label identifying control (unperturbed) cells.
    top_k      : int — number of DEGs to keep per perturbation (default 20).

    Returns
    -------
    dict mapping pert_label (str) → {
        "mask"       : (G,) bool Tensor — True at top-k DEG positions,
        "indices"    : (top_k,) long Tensor — gene indices (descending |Δ|),
        "ctrl_mean"  : (G,) float32 Tensor — control mean expression,
        "pert_mean"  : (G,) float32 Tensor — perturbed mean expression,
    }
    Control cells are excluded from the returned table.

    Notes
    -----
    - If a perturbation has zero cells or fewer than 1 cell, it is skipped
      with a warning.
    - top_k is clamped to min(top_k, G).
    """
    perts_arr = np.asarray(perts)
    counts_arr = np.asarray(counts, dtype=np.float32)
    n_genes = counts_arr.shape[1]
    top_k = min(top_k, n_genes)

    ctrl_mask = perts_arr == ctrl_label
    if ctrl_mask.sum() == 0:
        raise ValueError(
            f"No cells found with ctrl_label={ctrl_label!r}. "
            "Check that ctrl_label matches your data."
        )
    ctrl_mean = counts_arr[ctrl_mask].mean(axis=0)  # (G,)
    ctrl_mean_t = torch.from_numpy(ctrl_mean)

    unique_perts = sorted(set(perts_arr.tolist()) - {ctrl_label})
    table: dict[str, dict] = {}

    for pert in unique_perts:
        pert_mask = perts_arr == pert
        n_cells = pert_mask.sum()
        if n_cells < 1:
            logger.warning("Perturbation %r has 0 cells — skipping.", pert)
            continue

        pert_mean = counts_arr[pert_mask].mean(axis=0)  # (G,)
        delta = np.abs(pert_mean - ctrl_mean)  # (G,) absolute difference

        # argsort ascending; take last top_k (highest |Δ|), reverse to descending
        sorted_idx = np.argsort(delta)
        top_indices = sorted_idx[-top_k:][::-1].copy()  # (top_k,) descending

        mask = torch.zeros(n_genes, dtype=torch.bool)
        mask[top_indices] = True

        table[pert] = {
            "mask": mask,
            "indices": torch.from_numpy(top_indices),
            "ctrl_mean": ctrl_mean_t,
            "pert_mean": torch.from_numpy(pert_mean.copy()),
        }

    logger.info(
        "Built DEG table: %d perturbations, top_k=%d, n_genes=%d",
        len(table),
        top_k,
        n_genes,
    )
    return table
