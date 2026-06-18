import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from sc_evae.config.models import (
    sc_evaeConfig,
    FSQConfig,
    LFQConfig,
    NoQuantConfig,
    QuantizerConfig,
    VQConfig,
)
from vector_quantize_pytorch import VectorQuantize, FSQ, LFQ
from sc_evae.models.common import (
    GeneCountEmbedding,
    extract_pert_cond_input,
    make_pert_cond_module,
)
from sc_evae.models.transformer import build_backbone
from sc_evae.models.output_heads import NBHead, make_output_head

# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------


@dataclass
class QuantizerOutput:
    codes: torch.Tensor
    indices: torch.Tensor
    aux_losses: dict[str, torch.Tensor] = field(default_factory=dict)
    aux_metrics: dict[str, torch.Tensor] = field(default_factory=dict)


class Quantizer(nn.Module):
    def __init__(self, config: QuantizerConfig):
        super().__init__()
        self._codebook_size = getattr(config, "codebook_size", 0)
        if isinstance(config, VQConfig):
            self._type = "vq"
            self._codebook_dim = config.codebook_dim
            self._impl = VectorQuantize(
                dim=config.codebook_dim,
                codebook_size=config.codebook_size,
                commitment_weight=config.commitment_weight,
            )
        elif isinstance(config, FSQConfig):
            self._type = "fsq"
            self._codebook_dim = config.codebook_dim
            self._impl = FSQ(dim=config.codebook_dim, levels=config.fsq_level)
        elif isinstance(config, LFQConfig):
            self._type = "lfq"
            self._codebook_dim = config.codebook_dim
            self._impl = LFQ(
                codebook_size=config.codebook_size,
                entropy_loss_weight=config.entropy_loss_weight,
                commitment_loss_weight=0.0,
            )
        elif isinstance(config, NoQuantConfig):
            self._type = "noquant"
            self._codebook_dim = config.latent_dim
            self._beta_kl = config.beta_kl
            self.reparam_proj = nn.Linear(config.latent_dim, 2 * config.latent_dim)
        else:
            raise TypeError(f"Unknown quantizer config type: {type(config).__name__}")

    @property
    def codebook_dim(self) -> int:
        return self._codebook_dim

    def _compute_codebook_metrics(
        self, indices: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            flat_idx = indices.detach().reshape(-1).long()
            if flat_idx.numel() == 0:
                return {}
            counts = torch.bincount(flat_idx, minlength=self._codebook_size).float()
            used = counts > 0
            probs = counts / counts.sum().clamp_min(1.0)
            used_probs = probs[used]
            entropy = -(used_probs * used_probs.clamp_min(1e-12).log()).sum()
            return {
                "codebook_usage_frac": used.float().mean(),
                "codebook_perplexity": entropy.exp(),
            }

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Map integer indices (any shape) → code vectors (..., d_z)."""
        if self._type == "vq":
            return self._impl.get_codes_from_indices(indices)
        elif self._type == "fsq":
            return self._impl.indices_to_codes(indices)
        elif self._type == "lfq":
            return self._impl.get_codes_from_indices(indices)
        raise TypeError("decode_indices not supported for NoQuantConfig")

    def all_codes(self, device: torch.device) -> torch.Tensor:
        """Return full codebook matrix (vocab_size, d_z) for soft-code lookup."""
        idx = torch.arange(self._codebook_size, device=device)
        return self.decode_indices(idx.unsqueeze(0)).squeeze(0)

    def forward(self, x: torch.Tensor) -> QuantizerOutput:
        if self._type == "vq":
            codes, indices, commitment_loss = self._impl(x)
            return QuantizerOutput(
                codes,
                indices,
                aux_losses={"commitment_loss": commitment_loss},
                aux_metrics=self._compute_codebook_metrics(indices),
            )
        elif self._type == "fsq":
            codes, indices = self._impl(x)
            return QuantizerOutput(
                codes, indices, aux_metrics=self._compute_codebook_metrics(indices)
            )
        elif self._type == "lfq":
            codes, indices, entropy_loss = self._impl(x)
            return QuantizerOutput(
                codes,
                indices,
                aux_losses={"entropy_loss": entropy_loss},
                aux_metrics=self._compute_codebook_metrics(indices),
            )
        else:  # noquant — Gaussian reparameterization
            mu_logvar = self.reparam_proj(x)
            mu, log_var = mu_logvar.chunk(2, dim=-1)
            # Clamp mu and log_var for numerical stability.  scLDM's
            # ``GaussianLinearLayer`` only clamps log_scale to [-7, 5] (μ
            # unclamped) and uses a deterministic encoder by default; we add
            # μ ∈ [-4, 4] explicitly because under low β_kl the recon objective
            # does not bound μ and KL = 0.5·(μ² + σ² − log σ² − 1) then explodes
            # quadratically in μ once anything nudges the posterior.
            mu = mu.clamp(min=-4.0, max=4.0)
            log_var = log_var.clamp(min=-14.0, max=10.0)
            z = (
                mu + torch.randn_like(mu) * (0.5 * log_var).exp()
                if self.training
                else mu
            )
            # Per-cell KL (nats per cell): sum over latent dims (num_latents, d_latent),
            # mean over batch.  β_kl now directly multiplies per-cell nats — readable
            # as "KL budget per cell relative to recon" without needing to factor out
            # the latent shape.  Previous reduction .mean() averaged over all three
            # axes which made β_kl's effective weight depend on (num_latents · d_latent).
            kl_loss = (
                0.5 * (mu.pow(2) + log_var.exp() - log_var - 1).sum(dim=(-2, -1)).mean()
            )
            return QuantizerOutput(
                codes=z,
                indices=torch.zeros(z.shape[:2], dtype=torch.long, device=z.device),
                aux_losses={"kl_loss": self._beta_kl * kl_loss},
                aux_metrics={"kl_per_cell": kl_loss.detach()},
            )


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------
# TODO: add quantization tricks like kmeans, gradient-rotation, cosine-similarity, etc.


class sc_evae(nn.Module):
    def __init__(
        self,
        config: sc_evaeConfig,
        log_lib_prior_mu: float = 0.0,
        log_lib_prior_std: float = 1.0,
    ):
        super().__init__()
        self.config = config
        # Log-normal prior over library size — used at generation time to sample l_n
        # when no input cell is available.  Set from training-set statistics via
        # DataStats.log_lib_mean / log_lib_std; saved in the checkpoint as buffers.
        self.register_buffer("log_lib_prior_mu", torch.tensor(log_lib_prior_mu))
        self.register_buffer("log_lib_prior_std", torch.tensor(log_lib_prior_std))

        # ------------------------------------------------------------------
        # Count encoder (input embedding)
        # ------------------------------------------------------------------
        self.count_proj_in = GeneCountEmbedding(
            config.input_head, config.encoder.d_model
        )

        # ------------------------------------------------------------------
        # Output head — all projection + loss + predict logic lives here
        # ------------------------------------------------------------------
        self.output_head = make_output_head(
            config.output_head, config.decoder.d_model, num_genes=config.num_genes
        )

        # ------------------------------------------------------------------
        # Learnable latent / gene embeddings
        # ------------------------------------------------------------------
        self.latent_tokens = nn.Embedding(
            config.num_latent_tokens, config.encoder.d_model
        )
        self.gene_tokens = nn.Embedding(config.num_genes, config.encoder.d_model)
        nn.init.normal_(self.latent_tokens.weight, std=0.02)
        nn.init.normal_(self.gene_tokens.weight, std=0.02)
        if config.encoder.d_model != config.decoder.d_model:
            self.gene_tokens_proj = nn.Linear(
                config.encoder.d_model, config.decoder.d_model
            )
        else:
            self.gene_tokens_proj = nn.Identity()

        # ------------------------------------------------------------------
        # Quantizer + projections
        # ------------------------------------------------------------------
        self.quantizer = Quantizer(config.quantizer)
        self.quantizer_proj_in = nn.Linear(
            config.encoder.d_model, self.quantizer.codebook_dim
        )
        self.quantizer_proj_out = nn.Linear(
            self.quantizer.codebook_dim, config.decoder.d_model
        )

        # ------------------------------------------------------------------
        # Transformer backbones
        # ------------------------------------------------------------------
        self.encoder = build_backbone(config.encoder)
        self.decoder = build_backbone(config.decoder)

        # ------------------------------------------------------------------
        # Perturbation + cell-type conditioning (decoder-only)
        # ------------------------------------------------------------------
        if config.do_pert_cond or config.do_celltype_cond:
            _dit_types = ("dit", "dit_cross_attention")
            if config.decoder.type not in _dit_types:
                raise ValueError(
                    f"do_pert_cond / do_celltype_cond=True requires decoder.type in "
                    f"{_dit_types}, got {config.decoder.type!r}"
                )
        self.pert_proj_dec = make_pert_cond_module(config, config.decoder.d_model)
        if config.do_celltype_cond:
            if config.num_cell_types is None:
                raise ValueError(
                    "do_celltype_cond=True but config.num_cell_types is None — "
                    "ensure the dataset exposes num_cell_types."
                )
            self.celltype_embed_dec = nn.Embedding(
                config.num_cell_types, config.decoder.d_model
            )
            nn.init.normal_(self.celltype_embed_dec.weight, std=0.02)

        # ------------------------------------------------------------------
        # scANVI-style perturbation classifier head on the mean-pooled
        # pre-quantization encoder output (d_model-dim per cell). Encoder does
        # NOT see pert labels — the classifier reads them out.
        # ------------------------------------------------------------------
        self.pert_classifier: nn.Module | None = None
        if config.train_pert_classifier:
            if config.num_perts is None:
                raise ValueError(
                    "train_pert_classifier=True but config.num_perts is None."
                )
            dropout = config.encoder.attn_dropout
            layers: list[nn.Module] = []
            d = config.encoder.d_model
            for h in config.classifier_hidden:
                layers.append(nn.Linear(d, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                d = h
            layers.append(nn.Linear(d, config.num_perts))
            self.pert_classifier = nn.Sequential(*layers)

        # ------------------------------------------------------------------
        # ESM2 regressor head on the same mean-pooled encoder output. Trained
        # with cosine loss against batch["pert_emb"]; cells whose pert is
        # missing from pert_mapping have zero pert_emb and are masked out.
        # ------------------------------------------------------------------
        self.pert_regressor: nn.Module | None = None
        if config.train_pert_regressor:
            if config.pert_emb_dim is None:
                raise ValueError(
                    "train_pert_regressor=True but config.pert_emb_dim is None."
                )
            dropout = config.encoder.attn_dropout
            layers = []
            d = config.encoder.d_model
            for h in config.regressor_hidden:
                layers.append(nn.Linear(d, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                d = h
            layers.append(nn.Linear(d, config.pert_emb_dim))
            self.pert_regressor = nn.Sequential(*layers)

    def _run_encoder(
        self,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """Run the transformer encoder (always unconditional).

        Returns (B, num_latents, d_model).

        Perturbation conditioning is applied only to the decoder so that the
        encoder latent space captures data-driven structure without shortcuts
        through the conditioning signal.
        """
        batch_size = counts.shape[0]
        gene_ids = torch.arange(self.config.num_genes, device=counts.device)
        latent_ids = torch.arange(self.config.num_latent_tokens, device=counts.device)

        gene_embeds = self.gene_tokens(gene_ids).unsqueeze(0).expand(batch_size, -1, -1)
        counts_embeds = self.count_proj_in(counts, gene_embeds)
        x = self.latent_tokens(latent_ids).unsqueeze(0).expand(batch_size, -1, -1)
        x = self.encoder(x, counts_embeds)

        return x

    def encode(self, counts: torch.Tensor) -> torch.Tensor:
        """Encode counts → pre-quantization latent (B, num_latents, codebook_dim)."""
        return self.quantizer_proj_in(self._run_encoder(counts))

    def _build_decoder_cond(
        self,
        pert_emb: torch.Tensor | None,
        cell_type_idx: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Zero-init cond + pert projection + cell-type embedding (additive).

        Returns None when neither conditioning is enabled.
        """
        cfg = self.config
        if not (cfg.do_pert_cond or cfg.do_celltype_cond):
            return None
        cond = torch.zeros(batch_size, cfg.decoder.d_model, device=device)
        if cfg.do_pert_cond:
            if pert_emb is None:
                raise ValueError(
                    "do_pert_cond=True but pert-conditioning input is None in "
                    "_run_decoder() — batch missing 'pert_idx' (learned mode) "
                    "or 'pert_emb' (projected mode)."
                )
            if (
                cfg.pert_cond_source == "projected"
                and pert_emb.shape[-1] != cfg.pert_emb_dim
            ):
                raise ValueError(
                    f"pert_emb has dim {pert_emb.shape[-1]} but model was built "
                    f"with pert_emb_dim={cfg.pert_emb_dim}. "
                    "Ensure the pert_mapping embeddings match the model config."
                )
            cond = cond + self.pert_proj_dec(pert_emb)
        if cfg.do_celltype_cond:
            if cell_type_idx is None:
                raise ValueError(
                    "do_celltype_cond=True but cell_type_idx is None in _run_decoder()."
                )
            cond = cond + self.celltype_embed_dec(cell_type_idx)
        return cond

    def _run_decoder(
        self,
        codes: torch.Tensor,
        pert_emb: torch.Tensor | None = None,
        cell_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the transformer decoder. Returns (B, G, d_model)."""
        batch_size = codes.shape[0]
        gene_ids = torch.arange(self.config.num_genes, device=codes.device)
        gene_embeds = self.gene_tokens_proj(
            self.gene_tokens(gene_ids).unsqueeze(0).expand(batch_size, -1, -1)
        )
        code_embeds = self.quantizer_proj_out(codes)

        cond = self._build_decoder_cond(
            pert_emb, cell_type_idx, batch_size, codes.device
        )
        if cond is None:
            return self.decoder(gene_embeds, code_embeds)
        return (
            self.decoder(gene_embeds, cond)
            if self.config.decoder.type == "dit"
            else self.decoder(gene_embeds, cond, code_embeds)
        )

    def _sample_library_size(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Sample l_n ~ LogNormal(log_lib_prior_mu, log_lib_prior_std), shape (B, 1)."""
        eps = torch.randn(batch_size, 1, device=device)
        return (self.log_lib_prior_mu + eps * self.log_lib_prior_std).exp()

    def _predict_head(
        self,
        dec_x: torch.Tensor,
        l_n: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Call output_head.predict, threading l_n for NBHead."""
        if isinstance(self.output_head, NBHead):
            if l_n is None:
                l_n = self._sample_library_size(dec_x.shape[0], dec_x.device)
            return self.output_head.predict(dec_x, l_n)
        return self.output_head.predict(dec_x)

    def decode(
        self,
        codes: torch.Tensor,
        pert_emb: torch.Tensor | None = None,
        cell_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Decode quantized codes → predicted mean expression (B, G).

        Convenience method for generation; no loss is computed.
        l_n is sampled from the log-normal prior stored on this model.
        """
        return self._predict_head(self._run_decoder(codes, pert_emb, cell_type_idx))

    def decode_observed(
        self,
        codes: torch.Tensor,
        pert_emb: torch.Tensor | None,
        l_n: torch.Tensor,
        cell_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode codes using an observed per-cell library size (B, 1).

        Used by prior-model proxies to match the ``vae.pseudo_sample`` convention
        of conditioning on the input cell's library size rather than sampling
        from the log-normal prior.
        """
        return self._predict_head(
            self._run_decoder(codes, pert_emb, cell_type_idx), l_n=l_n
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        counts = batch["counts"]
        pert_emb = extract_pert_cond_input(batch, self.config)
        cell_type_idx = (
            batch.get("cell_type_idx") if self.config.do_celltype_cond else None
        )

        # Split the encode() pipeline so we can tap the pre-projection encoder
        # output for the classifier head (d_model-dim, richer than the
        # post-projection codebook_dim-dim representation).
        enc_out = self._run_encoder(counts)  # (B, L, d_model)
        x = self.quantizer_proj_in(enc_out)  # (B, L, codebook_dim)
        q_out = self.quantizer(x)
        dec_x = self._run_decoder(q_out.codes, pert_emb, cell_type_idx)

        recon_loss, head_aux = self.output_head(dec_x, counts)

        aux_losses = {**q_out.aux_losses, **head_aux}
        loss = recon_loss + sum(aux_losses.values(), start=recon_loss.new_zeros(()))

        extra_metrics: dict[str, torch.Tensor] = {}
        # Shared pooled representation for classifier and regressor heads.
        if self.pert_classifier is not None or self.pert_regressor is not None:
            pooled = enc_out.mean(dim=1)  # (B, d_model)

        # scANVI-style classifier head — CE on pert_idx from the mean-pooled
        # pre-quantization latent.
        if self.pert_classifier is not None:
            pert_idx = batch["pert_idx"]
            logits = self.pert_classifier(pooled)
            clf_nll = F.cross_entropy(logits, pert_idx)
            clf_acc = (logits.argmax(dim=-1) == pert_idx).float().mean()
            loss = loss + self.config.classifier_weight * clf_nll
            extra_metrics["clf_nll"] = clf_nll.detach()
            extra_metrics["clf_acc"] = clf_acc.detach()

        # ESM2 regressor head — cosine loss on pert_emb; mask zero targets.
        if self.pert_regressor is not None and "pert_emb" in batch:
            target = batch["pert_emb"]  # (B, D_esm)
            pred = self.pert_regressor(pooled)  # (B, D_esm)
            valid = target.norm(dim=-1) > 1e-6  # (B,)
            n_valid = valid.sum()
            if n_valid > 0:
                cos = F.cosine_similarity(pred[valid], target[valid], dim=-1)
                reg_loss = (1.0 - cos).mean()
            else:
                reg_loss = pred.new_zeros(())
            loss = loss + self.config.regressor_weight * reg_loss
            extra_metrics["reg_cos_loss"] = reg_loss.detach()
            extra_metrics["reg_valid_frac"] = (
                n_valid.float() / target.shape[0]
            ).detach()

        return loss, {
            "recon_loss": recon_loss,
            **aux_losses,
            **q_out.aux_metrics,
            **extra_metrics,
        }

    @torch.no_grad()
    def pseudo_sample(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode → quantize → decode → predicted mean expression (B, G).

        Uses the observed library size from the input cell (not sampled from prior).
        """
        counts = batch["counts"]
        pert_emb = extract_pert_cond_input(batch, self.config)
        cell_type_idx = (
            batch.get("cell_type_idx") if self.config.do_celltype_cond else None
        )
        x = self.encode(counts)
        q_out = self.quantizer(x)
        l_n = counts.sum(dim=-1, keepdim=True)
        return self._predict_head(
            self._run_decoder(q_out.codes, pert_emb, cell_type_idx), l_n=l_n
        )
