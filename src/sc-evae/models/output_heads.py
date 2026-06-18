"""
Output heads and input embedding factory for sc_evae.

Each output head receives transformer decoder output x: (B, G, d_model) and
handles the final projection, loss computation (forward), and inference (predict).

  forward(x, counts) -> (recon_loss, aux_losses)   — training
  predict(x)         -> (B, G)                      — inference (predicted mean)

Factory functions use isinstance dispatch on the ChoiceRegistry config objects —
never string comparison on a type field.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sc_evae.config.models import (
    CrossEntropyHeadConfig,
    HurdleHeadConfig,
    MSEHeadConfig,
    NBHeadConfig,
    OutputHeadConfig,
)
from sc_evae.models.losses import (
    cross_entropy_loss,
    hurdle_loss,
    nb_loss,
)

# ---------------------------------------------------------------------------
# Continuous-data heads (is_count_discrete=False, log1p-normalised input)
# ---------------------------------------------------------------------------


class MSEHead(nn.Module):
    """Direct regression: single scalar per gene."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(
        self, x: torch.Tensor, counts: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return F.mse_loss(self.proj(x).squeeze(-1), counts), {}

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).squeeze(-1)


class HurdleHead(nn.Module):
    """Binary zero/nonzero classifier + MSE on nonzero entries."""

    def __init__(self, d_model: int, config: HurdleHeadConfig):
        super().__init__()
        self.proj = nn.Linear(d_model, 2)
        self.weight_for_bce_loss = config.weight_for_bce_loss

    def forward(
        self, x: torch.Tensor, counts: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return (
            hurdle_loss(
                self.proj(x), counts, weight_for_bce_loss=self.weight_for_bce_loss
            ),
            {},
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # Sample the zero/nonzero gate per (cell, gene), then mask mu.
        # Matches the "one draw per cell" convention used by CrossEntropyHead
        # and is faithful to the hurdle generative model: predictions inherit
        # the data's sparsity instead of returning a flat magnitude on
        # likely-zero genes (where mu has no gradient signal at training time).
        out = self.proj(x)  # (B, G, 2)
        logit_p0 = out[..., 0]
        mu = out[..., 1]
        p_nonzero = (1.0 - torch.sigmoid(logit_p0.float())).clamp(0.0, 1.0)
        is_nonzero = torch.bernoulli(p_nonzero).to(mu.dtype)  # (B, G)
        return is_nonzero * mu


# ---------------------------------------------------------------------------
# Discrete-data heads (is_count_discrete=True, raw integer counts)
# ---------------------------------------------------------------------------


class NBHead(nn.Module):
    """
    Negative Binomial parametric head with library-size factorisation,
    matching scLDM's ``NegativeBinomialTransformerLayer`` (shared_theta mode).

    The NB mean is parameterised as ``mu = l_n * rho`` where:
      - ``rho`` is the softmax across genes of a single scalar per gene
        (constrained to the probability simplex — relative expression
        fractions).  Per-gene tokens are independent upstream; the softmax
        at the output binds them into a valid distribution.
      - ``l_n`` is the per-cell library size (total UMI count).

    Dispersion ``theta`` is **per-gene, cell-independent**: a learnable
    ``nn.Embedding(num_genes, 1)`` initialised to ones, passed through
    ``exp`` for positivity.  This matches scLDM's default
    ``shared_theta=True`` and scvi-style parameterisation — more stable
    than per-cell dispersion and gives each gene its own dispersion prior.
    """

    def __init__(
        self,
        d_model: int,
        num_genes: int,
        temperature: float = 1.0,
        min_theta: float = 1e-2,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.temperature = temperature
        self.min_theta = min_theta
        self.params = nn.Linear(d_model, 1)  # rho logits per gene
        self.theta = nn.Embedding(num_genes, 1)  # per-gene log-dispersion
        nn.init.ones_(self.theta.weight)

    def _theta(self, device: torch.device) -> torch.Tensor:
        gene_ids = torch.arange(self.num_genes, device=device)
        return torch.exp(self.theta(gene_ids).squeeze(-1)).clamp(min=self.min_theta)

    def _rho(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.params(x).squeeze(-1)  # (B, G)
        return F.softmax(logits / self.temperature, dim=1)  # (B, G)

    def forward(
        self, x: torch.Tensor, counts: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rho = self._rho(x)  # (B, G)
        theta = self._theta(x.device)  # (G,)
        l_n = counts.sum(dim=-1, keepdim=True)  # (B, 1)
        mu = l_n * rho  # (B, G)
        return nb_loss(mu, theta, counts), {}

    def predict(
        self, x: torch.Tensor, l_n: torch.Tensor, sample: bool = True
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Decoder output (B, G, d_model).
        l_n:
            Library size per cell (B, 1). At generation time, sample from
            ``LogNormal(log_lib_prior_mu, log_lib_prior_std)`` stored on the VAE.
            At reconstruction time, use ``counts.sum(dim=-1, keepdim=True)``.
        sample:
            If True (default), draw integer counts from ``NB(mu, theta)`` —
            matches scLDM's reconstruction path and lets ``theta`` shape the
            output. If False, return the deterministic mean ``mu = l_n * rho``.
        """
        mu = l_n * self._rho(x)  # (B, G)
        if not sample:
            return mu
        theta = self._theta(x.device).expand_as(mu)  # (B, G)
        mu_f, theta_f = mu.float(), theta.float()
        probs = mu_f / (mu_f + theta_f)
        nb = torch.distributions.NegativeBinomial(total_count=theta_f, probs=probs)
        return nb.sample().to(x.dtype)


class CrossEntropyHead(nn.Module):
    """Non-parametric categorical over [0, max_count]."""

    def __init__(self, d_model: int, config: CrossEntropyHeadConfig):
        super().__init__()
        self.proj = nn.Linear(d_model, config.max_count + 1)
        self.max_count = config.max_count

    def forward(
        self, x: torch.Tensor, counts: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return cross_entropy_loss(self.proj(x), counts, self.max_count), {}

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # Sample one bin index per (cell, gene) from the categorical.
        # Cast to float32 — multinomial does not support bf16.
        probs = self.proj(x).float().softmax(dim=-1)
        flat = probs.reshape(-1, probs.shape[-1])
        samples = torch.multinomial(flat, num_samples=1).squeeze(-1)
        return samples.reshape(probs.shape[:-1]).to(x.dtype)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_output_head(
    config: OutputHeadConfig,
    d_model: int,
    num_genes: int | None = None,
) -> nn.Module:
    """Construct the output head from a head config and the decoder d_model.

    ``num_genes`` is required for ``NBHead`` (per-gene dispersion embedding)
    and ignored by other heads.
    """
    if isinstance(config, MSEHeadConfig):
        return MSEHead(d_model)
    if isinstance(config, HurdleHeadConfig):
        return HurdleHead(d_model, config)
    if isinstance(config, NBHeadConfig):
        if num_genes is None:
            raise ValueError(
                "NBHead requires num_genes to build its per-gene dispersion "
                "embedding.  Pass num_genes to make_output_head."
            )
        return NBHead(
            d_model,
            num_genes,
            temperature=config.temperature,
            min_theta=config.min_theta,
        )
    if isinstance(config, CrossEntropyHeadConfig):
        return CrossEntropyHead(d_model, config)
    raise TypeError(f"Unknown OutputHeadConfig type: {type(config).__name__}")
