from sc_evae.metrics.perturbation import (
    pearson_delta_deg20,
    r2_deg20,
    batch_pearson_delta_deg20,
    batch_r2_deg20,
)
from sc_evae.metrics.deg_table import build_deg_table
from sc_evae.metrics.de_table import build_de_table, compute_pert_de_table
from sc_evae.metrics.distribution import (
    frechet_distance_pca,
    mmd2_rbf_pca,
    mmd2_rbf_scldm,
    wasserstein2_pca,
)
from sc_evae.metrics.state_metrics import (
    aggregate_per_pert,
    de_spearman_lfc_sig,
    de_spearman_sig,
    discrimination_score_l1,
    mae_log2fc_sig,
    overlap_at_n,
    pr_auc,
)

__all__ = [
    # Train-time proxy metrics (per-batch single-cell; logged as proxy_* during
    # training — NOT the paper's reported numbers; see scripts/eval.py for those)
    "pearson_delta_deg20",
    "r2_deg20",
    "batch_pearson_delta_deg20",
    "batch_r2_deg20",
    "build_deg_table",
    # State / cell-eval-faithful metrics
    "build_de_table",
    "compute_pert_de_table",
    "discrimination_score_l1",
    "overlap_at_n",
    "de_spearman_lfc_sig",
    "de_spearman_sig",
    "pr_auc",
    "mae_log2fc_sig",
    "aggregate_per_pert",
    # scLDM distribution-level generation metrics
    "wasserstein2_pca",
    "mmd2_rbf_pca",
    "mmd2_rbf_scldm",
    "frechet_distance_pca",
]
