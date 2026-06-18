"""
src/sc_evae/models/scvi.py
------------------------
scVI (Lopez et al., Nature Methods 2018) — MLP-based VAE with negative binomial
reconstruction likelihood and factored observed library size.

Design summary
~~~~~~~~~~~~~~
* Encoder: MLP(log1p(x)) → (z_mu, z_logvar) → z via reparameterization.
* Decoder: MLP(z) → per-gene logits; softmax yields gene *proportions* ρ that
  sum to 1 across G. The NB mean is ``μ_g = ℓ · ρ_g`` where ``ℓ`` is the
  observed per-cell library size ``sum(x, dim=-1)``. This decoupling of
  "which genes" (ρ) from "how deep" (ℓ) is scVI's signature trick.
* Dispersion θ is a learned per-gene vector (canonical "dispersion='gene'"
  mode). Parameterized as ``θ = exp(log_theta)`` for positivity.
* Loss: ``-E[NB(x | μ, θ)] + β · KL(q(z|x) || N(0, I))`` with linear β warmup.

Preprocessing contract
~~~~~~~~~~~~~~~~~~~~~~
YAML must set ``dataset.apply_normalize=false``, ``dataset.apply_log1p=false``,
``dataset.use_binning="none"``. The raw counts arrive as ``batch["counts"]``
(float32). log1p is applied inside ``encode`` before the encoder MLP; the raw
counts are used as the NB target and as the library-size source.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sc_evae.config.models import ScVIConfig

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _FCLayers(nn.Module):
    """Stack of (Linear → BatchNorm1d → ReLU → Dropout) blocks."""

    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            # Momentum / eps match scvi-tools defaults.
            layers.append(nn.BatchNorm1d(h, momentum=0.01, eps=1e-3))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Encoder(nn.Module):
    def __init__(self, n_in: int, hidden: list[int], n_latent: int, dropout: float):
        super().__init__()
        self.fc = _FCLayers(n_in, hidden, dropout)
        self.mu = nn.Linear(self.fc.out_dim, n_latent)
        self.logvar = nn.Linear(self.fc.out_dim, n_latent)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        z_mu = self.mu(h)
        # Clamp logvar for numerical safety (matches scvi-tools [-10, 10]).
        z_logvar = self.logvar(h).clamp(min=-10.0, max=10.0)
        return z_mu, z_logvar


class _Decoder(nn.Module):
    def __init__(self, n_latent: int, hidden: list[int], n_genes: int, dropout: float):
        super().__init__()
        self.fc = _FCLayers(n_latent, hidden, dropout)
        self.rho_logits = nn.Linear(self.fc.out_dim, n_genes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.rho_logits(self.fc(z))


# ---------------------------------------------------------------------------
# NB log-probability (Gamma–Poisson parameterization; matches scvi-tools)
# ---------------------------------------------------------------------------


def _nb_log_prob(
    x: torch.Tensor,  # (B, G) observed counts
    mu: torch.Tensor,  # (B, G) NB mean (> 0)
    theta: torch.Tensor,  # (G,) or (B, G) NB inverse dispersion (> 0)
    eps: float = 1e-8,
) -> torch.Tensor:
    """Element-wise log NB(x | μ, θ). Returns tensor of shape (B, G).

    Uses the Gamma–Poisson parameterization:
        p(x | μ, θ) = Γ(x+θ)/(Γ(θ)·x!) · (θ/(θ+μ))^θ · (μ/(θ+μ))^x
    """
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)
    log_theta_mu = torch.log(theta + mu + eps)
    return (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta + eps) - log_theta_mu)
        + x * (torch.log(mu + eps) - log_theta_mu)
    )


# ---------------------------------------------------------------------------
# ScVI module
# ---------------------------------------------------------------------------


class ScVI(nn.Module):
    """scVI with observed library size and per-gene NB dispersion."""

    def __init__(
        self,
        config: ScVIConfig,
        log_lib_prior_mu: float = 0.0,
        log_lib_prior_std: float = 1.0,
    ):
        super().__init__()
        self.config = config

        # Buffers carried for API parity with sc_evae (unused in
        # observed-library mode, kept so checkpoints remain portable if we
        # later add a latent-ℓ variant).
        self.register_buffer(
            "log_lib_prior_mu", torch.tensor(log_lib_prior_mu, dtype=torch.float32)
        )
        self.register_buffer(
            "log_lib_prior_std", torch.tensor(log_lib_prior_std, dtype=torch.float32)
        )

        if config.num_genes is None:
            raise ValueError("ScVI requires config.num_genes to be resolved.")

        self.encoder = _Encoder(
            n_in=config.num_genes,
            hidden=config.encoder_hidden,
            n_latent=config.latent_dim,
            dropout=config.dropout_rate,
        )
        self.decoder = _Decoder(
            n_latent=config.latent_dim,
            hidden=config.decoder_hidden,
            n_genes=config.num_genes,
            dropout=config.dropout_rate,
        )
        # Per-gene log-dispersion (inverse-dispersion parameter); θ_g = exp(log_theta_g).
        self.log_theta = nn.Parameter(torch.zeros(config.num_genes))

        # scANVI-style classifier head: q(pert_idx | z). Active only when
        # config.train_pert_classifier=True AND num_perts was resolved.
        self.classifier: nn.Module | None = None
        if config.train_pert_classifier:
            if config.num_perts is None:
                raise ValueError(
                    "train_pert_classifier=True but config.num_perts is None."
                )
            layers: list[nn.Module] = []
            d = config.latent_dim
            for h in config.classifier_hidden:
                layers.append(nn.Linear(d, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.dropout_rate))
                d = h
            layers.append(nn.Linear(d, config.num_perts))
            self.classifier = nn.Sequential(*layers)

        # ESM2-style regressor head: f(z) → R^pert_emb_dim, trained with cosine
        # loss against batch["pert_emb"]. Injects relational geometry from the
        # pert-feature space into z.
        self.pert_regressor: nn.Module | None = None
        if config.train_pert_regressor:
            if config.pert_emb_dim is None:
                raise ValueError(
                    "train_pert_regressor=True but config.pert_emb_dim is None."
                )
            layers = []
            d = config.latent_dim
            for h in config.regressor_hidden:
                layers.append(nn.Linear(d, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.dropout_rate))
                d = h
            layers.append(nn.Linear(d, config.pert_emb_dim))
            self.pert_regressor = nn.Sequential(*layers)

        # Step counter for KL warmup — buffer so it persists across checkpoints.
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long))

    # ------------------------------------------------------------------ API

    def encode(self, counts: torch.Tensor) -> torch.Tensor:
        """Return posterior mean ``z_mu`` of shape ``(B, latent_dim)``.

        Accepts raw counts ``(B, G)``; applies log1p internally.
        This is what ``scripts/eval_pfizer.py`` calls.
        """
        log_x = torch.log1p(counts.clamp_min(0.0))
        z_mu, _ = self.encoder(log_x)
        return z_mu

    @torch.no_grad()
    def pseudo_sample(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return NB posterior-mean expression ``(B, G)`` in raw-count scale.

        Library size is taken from the input cell (observed, not sampled).
        Used by ``run_validation`` for the DEG proxy metric.
        """
        counts = batch["counts"].clamp_min(0.0)
        lib = counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
        z_mu, _ = self.encoder(torch.log1p(counts))
        rho = F.softmax(self.decoder(z_mu), dim=-1)
        return lib * rho

    # --------------------------------------------------------- training path

    def _kl_weight_value(self) -> float:
        if self.config.kl_warmup_steps <= 0:
            return float(self.config.kl_weight)
        ratio = float(self._step.item()) / float(self.config.kl_warmup_steps)
        return float(self.config.kl_weight) * min(1.0, ratio)

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        counts = batch["counts"].clamp_min(0.0)
        lib = counts.sum(dim=-1, keepdim=True).clamp_min(1.0)  # (B, 1)
        log_x = torch.log1p(counts)

        z_mu, z_logvar = self.encoder(log_x)
        if self.training:
            z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mu)
        else:
            z = z_mu

        rho = F.softmax(self.decoder(z), dim=-1)  # (B, G) sum-to-1
        mu = lib * rho  # (B, G) NB mean
        theta = torch.exp(self.log_theta)  # (G,) > 0

        log_p = _nb_log_prob(counts, mu, theta)  # (B, G)
        recon_nll = -log_p.sum(dim=-1).mean()  # scalar: mean over cells

        # KL(q(z|x) || N(0, I)) summed over latent dims, averaged over cells.
        kl_z = -0.5 * (1.0 + z_logvar - z_mu.pow(2) - z_logvar.exp()).sum(dim=-1).mean()

        kl_weight = self._kl_weight_value()
        loss = recon_nll + kl_weight * kl_z

        metrics: dict[str, torch.Tensor] = {
            "recon_nll": recon_nll.detach(),
            "kl_z": kl_z.detach(),
            "kl_weight": torch.tensor(kl_weight, device=counts.device),
            "theta_mean": theta.mean().detach(),
            "theta_median": theta.median().detach(),
            "lib_mean": lib.mean().detach(),
        }

        # scANVI-style classifier: q(pert_idx | z) with CE loss.
        # Feed the SAME z the decoder consumes (sampled during training,
        # z_mu at eval) — consistent with scANVI's reparameterization policy.
        if self.classifier is not None:
            pert_idx = batch["pert_idx"]
            logits = self.classifier(z)
            clf_nll = F.cross_entropy(logits, pert_idx)
            clf_acc = (logits.argmax(dim=-1) == pert_idx).float().mean()
            loss = loss + self.config.classifier_weight * clf_nll
            metrics["clf_nll"] = clf_nll.detach()
            metrics["clf_acc"] = clf_acc.detach()

        # ESM2-style regressor: f(z) → pert_emb, cosine loss. Cells whose
        # pert is missing from pert_mapping have zero-vector pert_emb and
        # are masked out of the loss.
        if self.pert_regressor is not None and "pert_emb" in batch:
            target = batch["pert_emb"]  # (B, D_esm)
            pred = self.pert_regressor(z)  # (B, D_esm)
            valid = target.norm(dim=-1) > 1e-6  # (B,)
            n_valid = valid.sum()
            if n_valid > 0:
                cos = F.cosine_similarity(pred[valid], target[valid], dim=-1)
                reg_loss = (1.0 - cos).mean()
            else:
                reg_loss = pred.new_zeros(())
            loss = loss + self.config.regressor_weight * reg_loss
            metrics["reg_cos_loss"] = reg_loss.detach()
            metrics["reg_valid_frac"] = (n_valid.float() / target.shape[0]).detach()

        if self.training:
            self._step += 1

        return loss, metrics
