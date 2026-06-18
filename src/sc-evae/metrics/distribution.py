"""Distribution-level generation metrics in PCA space.

These mirror scLDM (Palla et al. 2025) Table 3 reporting on Replogle and
Parse1M:

  * **W2**          — 2-Wasserstein distance via optimal transport.
  * **MMD² RBF**    — squared MMD with Gaussian RBF kernel (median-bandwidth
                      heuristic by default).
  * **FD**          — Fréchet Distance, ``‖μ_r−μ_s‖² + Tr(Σ_r+Σ_s−2(Σ_rΣ_s)^½)``.

All three operate on PCA features (default 30 components) of log1p(CP10K)
cells. PCA is fit on the **true** cells (matches scLDM Appendix J.5: "We
compute the PCA on the true data, and project generated data using the
loadings"). Generated data is then projected through the same loadings.

Subsamples both groups to ``max_samples`` cells (default 5,000) before
computing — exact OT scales as O(n²·log n) and choke at ~10k cells per side.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _maybe_subsample(arr: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if arr.shape[0] <= max_samples:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=max_samples, replace=False)
    return arr[idx]


def _fit_pca_project(
    pred_cells: np.ndarray,
    true_cells: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA on ``true_cells`` and project both. Returns ``(pred_pc, true_pc)``.

    Centers using the true-data mean (transform's behavior). When
    ``true_cells.shape[0] < n_components``, falls back to ``min(n_obs, n_vars)``
    components and warns.
    """
    n_obs, n_vars = true_cells.shape
    k = min(n_components, n_obs - 1, n_vars)
    if k < n_components:
        logger.warning(
            "fit_pca: requested n_components=%d but n_obs=%d, n_vars=%d — "
            "using %d components.",
            n_components,
            n_obs,
            n_vars,
            k,
        )
    pca = PCA(n_components=k, svd_solver="auto", random_state=0)
    true_pc = pca.fit_transform(true_cells)
    pred_pc = pca.transform(pred_cells)
    return pred_pc.astype(np.float64), true_pc.astype(np.float64)


# ---------------------------------------------------------------------------
# Wasserstein-2
# ---------------------------------------------------------------------------


def wasserstein2_pca(
    pred_cells,
    true_cells,
    n_components: int = 30,
    method: str = "emd",
    sinkhorn_reg: float = 0.05,
    max_samples: int = 5_000,
    seed: int = 0,
) -> float:
    """2-Wasserstein distance between predicted and true cell distributions in
    PCA space.

    Mirrors ``scldm.evaluations.wasserstein`` with ``power=2``.

    ``method='emd'`` uses ``ot.emd2`` for exact optimal transport (slow:
    O(n³) for n ≤ ~5k); ``'sinkhorn'`` uses ``ot.sinkhorn2`` with entropy
    regularization (fast, gives a lower bound on true W2). Returns sqrt of
    OT cost (i.e., W2 distance, not squared).
    """
    import ot

    pred = _to_numpy(pred_cells).astype(np.float64)
    true = _to_numpy(true_cells).astype(np.float64)
    pred = _maybe_subsample(pred, max_samples, seed)
    true = _maybe_subsample(true, max_samples, seed + 1)

    pred_pc, true_pc = _fit_pca_project(pred, true, n_components)

    a = np.full(pred_pc.shape[0], 1.0 / pred_pc.shape[0])
    b = np.full(true_pc.shape[0], 1.0 / true_pc.shape[0])
    M = np.linalg.norm(pred_pc[:, None, :] - true_pc[None, :, :], axis=-1) ** 2

    if method == "emd":
        cost = ot.emd2(a, b, M, numItermax=10_000_000)
    elif method == "sinkhorn":
        cost = ot.sinkhorn2(a, b, M, reg=sinkhorn_reg, numItermax=10_000)
    else:
        raise ValueError(f"unknown method={method!r}; use 'emd' or 'sinkhorn'.")
    return float(np.sqrt(max(cost, 0.0)))


# ---------------------------------------------------------------------------
# MMD² with RBF kernel
# ---------------------------------------------------------------------------


def mmd2_rbf_pca(
    pred_cells,
    true_cells,
    n_components: int = 30,
    sigma: float | str = "median",
    biased: bool = False,
    max_samples: int = 5_000,
    seed: int = 0,
) -> float:
    """Squared MMD with Gaussian RBF kernel ``k(x,y)=exp(-‖x−y‖²/(2σ²))`` on
    PCA features.

    ``sigma='median'`` uses the median pairwise distance heuristic on pooled
    samples (most stable, no fixed scale). Pass a float to set σ explicitly.
    Returns ``MMD²`` (not the square root) — matches scLDM Table 3 column.

    Implementation follows the unbiased estimator in Gretton et al. 2012
    when ``biased=False`` (default). Set ``biased=True`` to match scLDM's
    ``MMDLoss`` which uses the biased ``mean(k_xx) + mean(k_yy) − 2 mean(k_xy)``.
    """
    pred = _to_numpy(pred_cells).astype(np.float64)
    true = _to_numpy(true_cells).astype(np.float64)
    pred = _maybe_subsample(pred, max_samples, seed)
    true = _maybe_subsample(true, max_samples, seed + 1)

    pred_pc, true_pc = _fit_pca_project(pred, true, n_components)

    # Squared pairwise distances on the pooled set, used both for σ heuristic
    # and the kernel evaluations.
    pooled = np.concatenate([pred_pc, true_pc], axis=0)
    sq_d_pooled = _sq_pdist(pooled)
    if isinstance(sigma, str) and sigma == "median":
        triu = np.triu_indices_from(sq_d_pooled, k=1)
        median_sq = float(np.median(sq_d_pooled[triu]))
        if median_sq < 1e-12:
            median_sq = 1.0
        gamma = 1.0 / (2.0 * median_sq)
    else:
        s = float(sigma)
        gamma = 1.0 / (2.0 * s * s)

    K_xx = np.exp(-gamma * _sq_pdist(pred_pc))
    K_yy = np.exp(-gamma * _sq_pdist(true_pc))
    K_xy = np.exp(-gamma * _sq_cdist(pred_pc, true_pc))

    m, n = K_xx.shape[0], K_yy.shape[0]
    if biased:
        return float(K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean())
    # Unbiased: drop diagonals from K_xx, K_yy.
    sum_xx = (K_xx.sum() - np.trace(K_xx)) / max(m * (m - 1), 1)
    sum_yy = (K_yy.sum() - np.trace(K_yy)) / max(n * (n - 1), 1)
    sum_xy = K_xy.mean()
    mmd2 = float(sum_xx + sum_yy - 2.0 * sum_xy)
    return max(mmd2, 0.0)


def _sq_pdist(x: np.ndarray) -> np.ndarray:
    n2 = (x * x).sum(axis=1, keepdims=True)
    return np.maximum(n2 + n2.T - 2.0 * (x @ x.T), 0.0)


def _sq_cdist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    nx = (x * x).sum(axis=1, keepdims=True)
    ny = (y * y).sum(axis=1, keepdims=True)
    return np.maximum(nx + ny.T - 2.0 * (x @ y.T), 0.0)


def mmd2_rbf_scldm(
    pred_cells,
    true_cells,
    n_components: int = 30,
    sigma_list: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0),
    biased: bool = True,
    max_samples: int = 5_000,
    seed: int = 0,
) -> float:
    """scLDM Table-3-style MMD² with a *mixture* of RBF kernels on PCA-30.

    Recipe (verified by cross-checking against scLDM Table 3 magnitudes
    on Parse 1M and Replogle):
      * PCA fit on the true cells, generated cells projected through the
        same loadings (paper Appendix J.5).
      * Kernel: sum of Gaussian RBFs with ``σ ∈ sigma_list``, i.e.
        ``K(x, y) = Σ_σ exp(-‖x − y‖²/(2σ²))``. The default
        ``[1, 2, 4, 8, 16]`` mirrors ``scldm.mmd.mix_rbf_mmd2`` (which is
        copied from atong01/conditional-flow-matching) — the multi-kernel
        recipe needed to reproduce Table 3's dynamic range (some baselines
        report MMD² > 2, exceeding the single-kernel max).
      * Estimator: biased by default (``mean(K_xx) + mean(K_yy) − 2 mean(K_xy)``)
        to match ``scldm.mmd._mmd2(..., biased=True)`` defaults. Set
        ``biased=False`` for the unbiased estimator written in the paper
        (Eq. 29) — the difference is negligible at the default
        ``max_samples=5000``.

    Notes
    -----
    The exact scLDM eval script is unreleased; the released
    ``scldm.evaluations.MMDLoss(RBFKernel(scale=1.0))`` is a single-kernel,
    no-PCA training-loop logger that saturates near 1/m on real data and
    is *not* what produces Table 3. This function mirrors what does:
    ``mix_rbf_mmd2`` from the same package, on PCA-30 features. Numbers
    line up with Table 3 to the right order of magnitude across baselines.

    Use ``mmd2_rbf_pca`` (median-heuristic σ + unbiased + PCA-30) for an
    independent, data-adaptive MMD that avoids the fixed-σ choice.
    """
    pred = _to_numpy(pred_cells).astype(np.float64)
    true = _to_numpy(true_cells).astype(np.float64)
    pred = _maybe_subsample(pred, max_samples, seed)
    true = _maybe_subsample(true, max_samples, seed + 1)

    pred_pc, true_pc = _fit_pca_project(pred, true, n_components)

    sq_xx = _sq_pdist(pred_pc)
    sq_yy = _sq_pdist(true_pc)
    sq_xy = _sq_cdist(pred_pc, true_pc)

    K_xx = np.zeros_like(sq_xx)
    K_yy = np.zeros_like(sq_yy)
    K_xy = np.zeros_like(sq_xy)
    for s in sigma_list:
        gamma = 1.0 / (2.0 * float(s) * float(s))
        K_xx += np.exp(-gamma * sq_xx)
        K_yy += np.exp(-gamma * sq_yy)
        K_xy += np.exp(-gamma * sq_xy)

    m, n = K_xx.shape[0], K_yy.shape[0]
    if biased:
        return float(K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean())
    sum_xx = (K_xx.sum() - np.trace(K_xx)) / max(m * (m - 1), 1)
    sum_yy = (K_yy.sum() - np.trace(K_yy)) / max(n * (n - 1), 1)
    sum_xy = K_xy.mean()
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


# ---------------------------------------------------------------------------
# Fréchet Distance (FID-style)
# ---------------------------------------------------------------------------


def frechet_distance_pca(
    pred_cells,
    true_cells,
    n_components: int = 30,
    eps: float = 1e-6,
    max_samples: int = 10_000,
    seed: int = 0,
) -> float:
    """Fréchet Distance between Gaussian fits in PCA space.

    Recipe from scLDM Appendix J.3:
      1. Stack and PCA-30 on TRUE only; project both.
      2. Estimate (μ, Σ) for both groups.
      3. ``FD = ‖μ_r − μ_s‖² + Tr(Σ_r + Σ_s − 2 (Σ_r·Σ_s)^½)``.

    The matrix square root is computed via ``scipy.linalg.sqrtm``; an
    ``eps·I`` regularizer is added to both covariances to avoid numerical
    issues with rank-deficient PCs. Returns FD (not its square root).
    """
    from scipy.linalg import sqrtm

    pred = _to_numpy(pred_cells).astype(np.float64)
    true = _to_numpy(true_cells).astype(np.float64)
    pred = _maybe_subsample(pred, max_samples, seed)
    true = _maybe_subsample(true, max_samples, seed + 1)

    pred_pc, true_pc = _fit_pca_project(pred, true, n_components)

    mu_p = pred_pc.mean(axis=0)
    mu_t = true_pc.mean(axis=0)
    cov_p = np.cov(pred_pc, rowvar=False) + eps * np.eye(pred_pc.shape[1])
    cov_t = np.cov(true_pc, rowvar=False) + eps * np.eye(true_pc.shape[1])

    diff = mu_p - mu_t
    covmean, _ = sqrtm(cov_p @ cov_t, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fd = diff @ diff + np.trace(cov_p) + np.trace(cov_t) - 2.0 * np.trace(covmean)
    return float(max(fd, 0.0))
