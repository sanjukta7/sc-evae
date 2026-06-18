import torch
import torch.nn.functional as F


def hurdle_loss(
    pred: torch.Tensor,
    counts: torch.Tensor,
    weight_for_bce_loss: float = 1.0,
) -> torch.Tensor:
    """
    Hurdle reconstruction loss for log1p-normalized data.

    Decoder outputs (B, G, 2): pred[:,:,0] = logit_p0 (zero-probability logit),
    pred[:,:,1] = mu (predicted positive mean).

    Loss = weight_for_bce_loss * BCE(logit_p0, is_zero)
           + MSE(mu, counts) on positive entries only.

    ~90% of single-cell entries are zero, so BCE on the full (B, G) grid is
    an easy signal and can dominate MSE (which averages over a much smaller
    nonzero support).  ``weight_for_bce_loss`` lets the caller rebalance —
    values < 1 downweight the (easy) zero head so gradient budget goes to
    the (harder) magnitude regression.

    Args:
        pred:                (B, G, 2) — logit_p0 and mu along the last dim.
        counts:              (B, G) — log1p-normalized UMI counts.
        weight_for_bce_loss: Multiplier on the BCE term.  Default 1.0 (equal
                             weight with the positive-only MSE term).

    Returns:
        Scalar loss.
    """
    logit_p0 = pred[:, :, 0]  # (B, G)
    mu = pred[:, :, 1]  # (B, G)

    is_zero = (counts == 0).float()
    bce = F.binary_cross_entropy_with_logits(logit_p0, is_zero, reduction="mean")
    n_nonzero = (1.0 - is_zero).sum().clamp_min(1.0)
    recon = ((1.0 - is_zero) * (mu - counts).pow(2)).sum() / n_nonzero
    return weight_for_bce_loss * bce + recon


def nb_loss(
    mu: torch.Tensor, theta: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    """
    Negative Binomial NLL for raw count data.

    Parameterization: NB(mu, theta) where mu = mean, theta = dispersion.
    As theta → ∞ the distribution converges to Poisson(mu).

    Reference formulation (same as scVI / scLDM):
      log P(x | mu, theta) = lgamma(x + theta) - lgamma(theta) - lgamma(x + 1)
                            + theta * log(theta / (theta + mu))
                            + x   * log(mu    / (mu    + theta))

    Reduction is **sum over genes, mean over batch** — matching scLDM
    (``-log_nb_positive(...).sum(dim=-1).mean()``).  Yields the proper
    per-cell NLL scale and keeps relative weighting against per-cell aux
    losses (commitment, KL, ...) meaningful.

    Args:
        mu:     (B, G) — predicted mean, strictly positive.  Broadcastable with theta.
        theta:  (G,) or (B, G) — dispersion parameter, strictly positive.
        counts: (B, G) — raw integer counts (float32).

    Returns:
        Scalar mean NLL (positive value — higher is worse).
    """
    eps = 1e-8
    # Defensive mu floor: with softmax-rho, mu_g = l_n * rho_g can drop to
    # 0 in fp32 when softmax is winner-take-all on some genes; if those
    # genes have nonzero counts, ``counts * log(mu)`` produces a -inf
    # gradient spike. The floor is well below any realistic prediction
    # (1e-8 corresponds to <1e-12 of typical libraries) so it only
    # activates when softmax has effectively collapsed.
    mu = mu.clamp(min=eps)
    log_theta_mu = torch.log(theta + mu + eps)
    nll = (
        -torch.lgamma(counts + theta)
        + torch.lgamma(theta)
        + torch.lgamma(counts + 1.0)
        - theta * (torch.log(theta + eps) - log_theta_mu)
        - counts * (torch.log(mu + eps) - log_theta_mu)
    )
    return nll.sum(dim=-1).mean()


def cross_entropy_loss(
    logits: torch.Tensor,
    counts: torch.Tensor,
    max_count: int,
) -> torch.Tensor:
    """
    Cross-entropy loss for raw discrete count data.

    Args:
        logits:    (B, G, vocab_size) — raw logits, vocab_size = max_count + 1.
        counts:    (B, G) — raw integer counts (float32).
        max_count: Maximum count value; counts above this are clamped.

    Returns:
        Scalar mean NLL.
    """
    B, G, V = logits.shape
    targets = counts.long().clamp(0, max_count)  # (B, G)
    return F.cross_entropy(logits.reshape(B * G, V), targets.reshape(B * G))
