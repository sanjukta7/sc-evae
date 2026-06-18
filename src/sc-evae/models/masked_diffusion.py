"""
Masked (absorbing) discrete diffusion prior.

Implements the MDLM / RADD family of absorbing-diffusion models.  The training
objective is the same in both variants (see docs/planning/absorbing_diffussion.md):

    L = E_{x, t, z_t}  [  (1 / lambda_t) *  sum_{i : z_t^i = m}  -log p_theta(x^i | z_t) ]

under the log-linear schedule  alpha_t = 1 - t,  lambda_t = t.

Variants:
  * MDLM  (time_conditioned=True, default) — adaLN-Zero DiT sees t.
  * RADD  (time_conditioned=False)         — t is not fed to the network.

Two operating modes:

  Mode A (vae_path set): a frozen sc_evae maps each cell to its (B, L)
    discrete codebook indices.  The denoiser learns p(x_0 | z_t^{UM}) over codes.
    pseudo_sample() runs one denoising pass from all-mask over codes → soft codes
    → VAE decode (a cheap proxy stand-in for the multi-step sample() below).

  Mode B (vae_path=None): the denoiser operates directly on (B, G) raw integer
    gene counts.  Each gene is a position with its own gene-identity embedding
    added to a learnable count-token embedding.  pseudo_sample() runs one denoising
    pass from all-mask and returns expected counts under the model posterior.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sc_evae.config.models import (
    CrossEntropyHeadConfig,
    MaskedDiffusionModelConfig,
    NoQuantConfig,
)
from sc_evae.models.common import (
    GeneCountEmbedding,
    extract_pert_cond_input,
    make_pert_cond_module,
)
from sc_evae.models.transformer import build_backbone

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for scalar time t in [0, 1] mapped to R^d."""

    def __init__(self, dim: int, scale: float = 1.0, max_period: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalTimeEmbedding requires even dim, got {dim}")
        self.dim = dim
        self.scale = float(scale)
        self.max_period = float(max_period)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0, 1]
        half = self.dim // 2
        device = t.device
        t_scaled = t.float() * self.scale
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * (
                -torch.log(torch.tensor(self.max_period, device=device))
                / max(half - 1, 1)
            )
        )
        args = t_scaled.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


def _low_discrepancy_uniform(
    batch_size: int,
    device: torch.device,
    eps: float,
) -> torch.Tensor:
    """MDLM App. D.3 low-discrepancy sampler: t_i = (i + u) / B, u ~ U[0,1]."""
    u = torch.rand((), device=device)
    idx = torch.arange(batch_size, device=device, dtype=torch.float32)
    t = (idx + u) / batch_size
    return t.clamp(eps, 1.0 - eps)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MaskedDiffusionModel(nn.Module):
    """
    Absorbing-diffusion / MDLM / RADD prior.

    Parameters
    ----------
    config : MaskedDiffusionModelConfig
    vae    : optional frozen sc_evae (Mode A).  When None → Mode B.
    """

    def __init__(
        self,
        config: MaskedDiffusionModelConfig,
        vae=None,  # sc_evae | None
    ):
        super().__init__()
        self.config = config
        d_model = config.transformer.d_model

        if vae is not None:
            # ------------------------------------------------------------------
            # Mode A: MDLM over VAE discrete latent codes
            # ------------------------------------------------------------------
            if isinstance(vae.config.quantizer, NoQuantConfig):
                raise ValueError(
                    "MaskedDiffusionModel requires a discrete VAE quantizer "
                    "(FSQConfig, VQConfig, or LFQConfig).  NoQuantConfig produces "
                    "continuous latents — use FlowMatchingModel instead."
                )
            vocab_size: int = vae.config.quantizer.codebook_size

            for p in vae.parameters():
                p.requires_grad_(False)
            vae.eval()
            self.vae = vae

            # MASK token id = vocab_size (one beyond the valid code range).
            self.mask_token_id: int = vocab_size
            self.vocab_size: int = vocab_size
            self.token_embed = nn.Embedding(vocab_size + 1, d_model)
            # Learnable positional embeddings for the L latent positions.
            self.pos_embed = nn.Embedding(vae.config.num_latent_tokens, d_model)
            nn.init.normal_(self.token_embed.weight, std=0.02)
            nn.init.normal_(self.pos_embed.weight, std=0.02)
            self._mode = "vae"
            # Output: vocab_size logits (never MASK — SUBS RB2).
            self.logits_proj = nn.Linear(d_model, vocab_size)
        else:
            # ------------------------------------------------------------------
            # Mode B: MDLM over raw integer gene counts
            # ------------------------------------------------------------------
            self.vae = None
            if not isinstance(config.output_head, CrossEntropyHeadConfig):
                raise TypeError(
                    "MaskedDiffusionModel (Mode B) requires a CrossEntropyHeadConfig "
                    f"output head, got {type(config.output_head).__name__}"
                )
            max_count = config.output_head.max_count
            vocab_size = max_count + 1
            # ``mask_token_id`` is still exposed for diagnostics / tests even
            # though Mode B no longer routes MASK through an embedding lookup.
            self.mask_token_id: int = vocab_size  # one beyond [0, max_count]
            self.vocab_size: int = vocab_size
            self.max_count: int = max_count

            # Gene-identity embeddings (positional role — one per gene).
            self.gene_tokens = nn.Embedding(config.num_genes, d_model)
            nn.init.normal_(self.gene_tokens.weight, std=0.02)
            # Count → d_model projection (learned / sinusoidal / MLP).  Matches
            # the signature used by AR / flow_matching.
            self.count_proj_in = GeneCountEmbedding(config.input_head, d_model)
            # Separate learned sentinel for masked positions.  Combined with
            # ``gene_tokens`` at embed time via ``torch.where``.
            self.mask_embed = nn.Parameter(torch.randn(d_model) * 0.02)
            self._mode = "counts"
            self.logits_proj = nn.Linear(d_model, vocab_size)

        # ------------------------------------------------------------------
        # Time + perturbation conditioning (fed into DiT via adaLN)
        # ------------------------------------------------------------------
        # SinusoidalTimeEmbedding → MLP → d_model; zero-contribution when
        # time_conditioned=False (RADD).  We still build the module when
        # do_pert_cond=True so the cond vector has a defined shape.
        if config.time_conditioned:
            self.time_embed = SinusoidalTimeEmbedding(
                d_model,
                scale=config.time_embedding_scale,
                max_period=config.time_embedding_max_period,
            )
            self.time_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.time_embed = None
            self.time_mlp = None

        self.pert_proj = make_pert_cond_module(config, d_model)

        if config.do_celltype_cond:
            if config.num_cell_types is None:
                raise ValueError(
                    "do_celltype_cond=True but config.num_cell_types is None — "
                    "ensure the dataset exposes num_cell_types."
                )
            self.celltype_embed = nn.Embedding(config.num_cell_types, d_model)
            nn.init.normal_(self.celltype_embed.weight, std=0.02)
        else:
            self.celltype_embed = None

        # ------------------------------------------------------------------
        # Backbone (bidirectional — enforced by config.__post_init__).
        # ------------------------------------------------------------------
        self.transformer = build_backbone(config.transformer)

    # --------------------------------------------------------------------------
    # Shared helpers
    # --------------------------------------------------------------------------

    def _build_cond(
        self,
        t: torch.Tensor,
        pert_emb: torch.Tensor | None,
        cls_cond: torch.Tensor | None = None,
        cell_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Return the adaLN conditioning vector (B, d_model), or None."""
        if self.config.transformer.type != "dit":
            return None  # plain transformer path — no cond signal used
        d_model = self.config.transformer.d_model
        cond = t.new_zeros(t.shape[0], d_model)
        if self.time_mlp is not None:
            cond = cond + self.time_mlp(self.time_embed(t))
        if self.pert_proj is not None:
            if pert_emb is None:
                raise ValueError("do_pert_cond=True but batch has no 'pert_emb' key.")
            cond = cond + self.pert_proj(pert_emb)
        if self.celltype_embed is not None:
            if cell_type_idx is None:
                raise ValueError(
                    "do_celltype_cond=True but batch has no 'cell_type_idx' key."
                )
            cond = cond + self.celltype_embed(cell_type_idx)
        if cls_cond is not None:
            cond = cond + cls_cond
        return cond

    def _run_backbone(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.config.transformer.type == "dit":
            return self.transformer(x, cond)
        return self.transformer(x)

    def _apply_subs_rb2(self, logits: torch.Tensor) -> torch.Tensor:
        """Already absent: we only project to vocab_size (MASK excluded)."""
        return logits

    # --------------------------------------------------------------------------
    # Mode A: VAE latent codes
    # --------------------------------------------------------------------------

    def _forward_vae_mode(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict]:
        counts = batch["counts"]
        pert_emb = extract_pert_cond_input(batch, self.config)
        cell_type_idx = (
            batch.get("cell_type_idx") if self.config.do_celltype_cond else None
        )
        cls_cond = batch.get("cls_cond")

        with torch.no_grad():
            pre_quant = self.vae.encode(counts)
            q_out = self.vae.quantizer(pre_quant)
            targets = q_out.indices  # (B, L) int64

        return self._shared_forward(
            targets=targets,
            pert_emb=pert_emb,
            use_pos_embed=True,
            cls_cond=cls_cond,
            cell_type_idx=cell_type_idx,
        )

    # --------------------------------------------------------------------------
    # Mode B: raw gene counts
    # --------------------------------------------------------------------------

    def _forward_counts_mode(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict]:
        counts = batch["counts"]
        pert_emb = extract_pert_cond_input(batch, self.config)
        cell_type_idx = (
            batch.get("cell_type_idx") if self.config.do_celltype_cond else None
        )
        cls_cond = batch.get("cls_cond")
        targets = counts.long().clamp(0, self.max_count)  # (B, G)
        return self._shared_forward(
            targets=targets,
            pert_emb=pert_emb,
            use_pos_embed=False,
            cls_cond=cls_cond,
            cell_type_idx=cell_type_idx,
        )

    # --------------------------------------------------------------------------
    # Shared forward (same math for both modes after we have target ids)
    # --------------------------------------------------------------------------

    def _embed_vae(self, token_ids: torch.Tensor, use_pos_embed: bool) -> torch.Tensor:
        B, L = token_ids.shape
        x = self.token_embed(token_ids)  # (B, L, d_model)
        if use_pos_embed:
            pos = torch.arange(L, device=token_ids.device)
            x = x + self.pos_embed(pos).unsqueeze(0)
        return x

    def _embed_counts(self, counts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``counts`` (B, G) long/float in [0, max_count] · ``mask`` (B, G) bool."""
        B, G = counts.shape
        gene_ids = torch.arange(G, device=counts.device)
        gene_embeds = self.gene_tokens(gene_ids).unsqueeze(0).expand(B, -1, -1)
        count_x = self.count_proj_in(counts, gene_embeds)  # (B, G, D)
        mask_x = self.mask_embed.expand(B, G, -1) + gene_embeds  # (B, G, D)
        return torch.where(mask.unsqueeze(-1), mask_x, count_x)

    def _shared_forward(
        self,
        targets: torch.Tensor,
        pert_emb: torch.Tensor | None,
        use_pos_embed: bool,
        cls_cond: torch.Tensor | None = None,
        cell_type_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        B, L = targets.shape
        device = targets.device

        # Sample t and mask under log-linear schedule alpha_t = 1 - t.
        t = _low_discrepancy_uniform(B, device, self.config.time_epsilon)  # (B,)
        mask_prob = t  # lambda_t = 1 - alpha_t = t
        mask = torch.bernoulli(mask_prob.unsqueeze(-1).expand(B, L)).bool()  # (B, L)

        if self._mode == "vae":
            z_t = torch.where(mask, self.mask_token_id, targets)  # (B, L)
            x = self._embed_vae(z_t, use_pos_embed=use_pos_embed)
        else:
            x = self._embed_counts(targets, mask)

        cond = self._build_cond(t, pert_emb, cls_cond, cell_type_idx=cell_type_idx)
        h = self._run_backbone(x, cond)  # (B, L, d_model)
        logits = self.logits_proj(h)  # (B, L, vocab_size)

        # Loss: (1/lambda) * CE on masked positions.  ``reduction="none"``
        # so we can reweight per-sample and restrict to masked positions.
        log_probs = F.log_softmax(logits, dim=-1)  # (B, L, vocab_size)
        gold = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, L)

        weight = 1.0 / mask_prob.clamp_min(self.config.time_epsilon)  # (B,)
        # Normalise per-sample by sequence length so the effective scale is
        # comparable to the AR cross-entropy (which is a per-token mean).
        per_sample = -(gold * mask.float()).sum(dim=-1) * weight / float(L)
        loss = per_sample.mean()

        with torch.no_grad():
            num_masked = mask.float().sum(dim=-1)  # (B,)
            metrics = {
                "mask_frac": mask.float().mean().detach(),
                "t_mean": t.mean().detach(),
                "num_masked": num_masked.mean().detach(),
            }
        return loss, metrics

    # --------------------------------------------------------------------------
    # Public API — matches the AutoregressiveModel / sc_evae interface
    # --------------------------------------------------------------------------

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        if self.vae is not None:
            return self._forward_vae_mode(batch)
        return self._forward_counts_mode(batch)

    def _pseudo_sample_vae_mode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Mode A proxy: single denoising pass from all-MASK over the L latent code
        positions, then decode through the VAE using soft codes (expected code vector
        under the model posterior).  Uses observed library size from the batch.
        """
        counts = batch["counts"]
        pert_emb = extract_pert_cond_input(batch, self.config)
        cell_type_idx = (
            batch.get("cell_type_idx") if self.config.do_celltype_cond else None
        )
        cls_cond = batch.get("cls_cond")
        B = counts.shape[0]
        device = counts.device
        L = self.vae.config.num_latent_tokens

        z = torch.full((B, L), self.mask_token_id, dtype=torch.long, device=device)
        x = self._embed_vae(z, use_pos_embed=True)
        t = torch.full((B,), 1.0 - self.config.time_epsilon, device=device)
        cond = self._build_cond(t, pert_emb, cls_cond, cell_type_idx=cell_type_idx)
        h = self._run_backbone(x, cond)
        logits = self.logits_proj(h)  # (B, L, vocab_size)

        probs = logits.softmax(dim=-1)  # (B, L, vocab_size)
        codebook = self.vae.quantizer.all_codes(device)  # (vocab_size, d_z)
        soft_codes = probs @ codebook  # (B, L, d_z)

        pert_emb_dec = extract_pert_cond_input(batch, self.vae.config)
        cell_type_idx_dec = (
            batch.get("cell_type_idx") if self.vae.config.do_celltype_cond else None
        )
        return self.vae.decode_observed(
            soft_codes,
            pert_emb_dec,
            counts.sum(-1, keepdim=True),
            cell_type_idx=cell_type_idx_dec,
        )

    @torch.no_grad()
    def pseudo_sample(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Return predicted mean expression (B, G) for proxy validation metrics.

        Mode A: single denoising from all-MASK over latent codes → soft codes → VAE decode.

        Mode B: starts from all-MASK, runs a single denoising pass to get the
        model's posterior over each gene count, and returns the posterior mean
        E[x | all-mask].  This is a cheap, teacher-free proxy — adequate for
        Pearson-DEG-20 style metrics that only care about relative magnitudes.
        """
        if self.vae is not None:
            return self._pseudo_sample_vae_mode(batch)

        counts = batch["counts"]
        pert_emb = extract_pert_cond_input(batch, self.config)
        cell_type_idx = (
            batch.get("cell_type_idx") if self.config.do_celltype_cond else None
        )
        cls_cond = batch.get("cls_cond")
        B, G = counts.shape
        device = counts.device

        # All-mask input: actual count values are ignored at masked positions.
        counts_zero = torch.zeros((B, G), dtype=torch.long, device=device)
        all_mask = torch.ones((B, G), dtype=torch.bool, device=device)
        x = self._embed_counts(counts_zero, all_mask)
        t = torch.full((B,), 1.0 - self.config.time_epsilon, device=device)
        cond = self._build_cond(t, pert_emb, cls_cond, cell_type_idx=cell_type_idx)
        h = self._run_backbone(x, cond)
        logits = self.logits_proj(h)  # (B, G, vocab_size)
        probs = logits.softmax(dim=-1)  # (B, G, vocab_size)
        vocab = torch.arange(self.vocab_size, device=device, dtype=probs.dtype)
        return (probs * vocab).sum(dim=-1)  # (B, G)

    # --------------------------------------------------------------------------
    # Sampling — ancestral reverse unmasking.  Not called from train loop; kept
    # here for completeness so `pseudo_sample` can be kept cheap.
    # --------------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        num_steps: int | None = None,
        pert_emb: torch.Tensor | None = None,
        device: torch.device | str = "cpu",
        cls_cond: torch.Tensor | None = None,
        cell_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate `(batch_size, seq_len)` samples starting from all-MASK.

        ``seq_len`` is taken from the model: VAE-mode uses
        ``vae.config.num_latent_tokens``; counts-mode uses ``config.num_genes``.

        Returns integer token ids in [0, vocab_size).  Uses uniform t-grid and
        the analytic reverse transition under the log-linear schedule
        (alpha_s - alpha_t) / (1 - alpha_t) per MDLM Eq. 6 / RADD App. D.
        """
        num_steps = num_steps or self.config.num_sample_steps
        device = torch.device(device)
        # Mode A tracks z as integer token ids with MASK sentinel.
        # Mode B tracks (counts, is_mask) — counts at unmasked positions, all
        # True in is_mask means "still to reveal".
        is_vae = self._mode == "vae"
        seq_len = self.vae.config.num_latent_tokens if is_vae else self.config.num_genes
        if is_vae:
            z = torch.full(
                (batch_size, seq_len),
                self.mask_token_id,
                dtype=torch.long,
                device=device,
            )
        else:
            counts = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
            is_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        # Reverse-time grid: t_T = 1, ..., t_0 = 0.
        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        for step_idx in range(num_steps):
            t_cur = ts[step_idx].clamp(self.config.time_epsilon, 1.0)
            t_nxt = ts[step_idx + 1].clamp_min(0.0)
            alpha_cur = 1.0 - t_cur
            alpha_nxt = 1.0 - t_nxt
            # Per-token unmask probability for this step.
            p_unmask = ((alpha_nxt - alpha_cur) / (1.0 - alpha_cur)).clamp(0.0, 1.0)

            if is_vae:
                x = self._embed_vae(z, use_pos_embed=True)
            else:
                x = self._embed_counts(counts, is_mask)
            t_batch = t_cur.expand(batch_size)
            cond = self._build_cond(
                t_batch, pert_emb, cls_cond, cell_type_idx=cell_type_idx
            )
            h = self._run_backbone(x, cond)
            logits = self.logits_proj(h)  # (B, L, vocab_size)
            probs = logits.softmax(dim=-1)

            # Sample replacement tokens from the model posterior.
            flat_probs = probs.reshape(-1, self.vocab_size)
            sampled = torch.multinomial(flat_probs, num_samples=1).view(
                batch_size, seq_len
            )

            if is_vae:
                currently_mask = z == self.mask_token_id
                unmask_now = torch.rand_like(z, dtype=torch.float32) < p_unmask
                update = currently_mask & unmask_now
                z = torch.where(update, sampled, z)
            else:
                unmask_now = torch.rand_like(counts, dtype=torch.float32) < p_unmask
                update = is_mask & unmask_now
                counts = torch.where(update, sampled, counts)
                is_mask = is_mask & ~update

        # Any remaining masks at t_0=0: reveal with a final greedy pass.
        if is_vae:
            still_masked = z == self.mask_token_id
        else:
            still_masked = is_mask
        if still_masked.any():
            if is_vae:
                x = self._embed_vae(z, use_pos_embed=True)
            else:
                x = self._embed_counts(counts, is_mask)
            t_batch = torch.full((batch_size,), self.config.time_epsilon, device=device)
            cond = self._build_cond(
                t_batch, pert_emb, cls_cond, cell_type_idx=cell_type_idx
            )
            h = self._run_backbone(x, cond)
            logits = self.logits_proj(h)
            final = logits.argmax(dim=-1)
            if is_vae:
                z = torch.where(z == self.mask_token_id, final, z)
            else:
                counts = torch.where(is_mask, final, counts)
                is_mask = torch.zeros_like(is_mask)
        return z if is_vae else counts
