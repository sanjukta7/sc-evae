"""
Conditional flow-matching prior.

Straight-line interpolation between Gaussian noise ``x0 ~ N(0, I)`` and clean
data ``x1`` with constant target velocity ``v = x1 - x0``.  The denoiser is
trained to match the velocity field (MSE), and sampling is Euler integration
of ``dx/dt = v_theta(x_t, t)`` from ``t=0`` to ``t=1``.

Two operating modes:

  Mode A (vae_path set): a frozen NoQuantConfig VAE produces continuous latent
    codes of shape (B, L, d_z).  The flow model operates on z-space.
    pseudo_sample() runs a short (4-step Heun) integration in z-space → VAE
    decode (a cheap proxy stand-in for the multi-step sample() below).

  Mode B (vae_path=None): the flow model operates directly on log1p-normalized
    gene counts of shape (B, G).  Gene identity embeddings + FiLM of the noisy
    count give per-gene tokens.  pseudo_sample() runs a short (4-step Heun)
    integration from Gaussian noise to return generated log-expression values.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sc_evae.config.models import (
    FlowMatchingModelConfig,
    NoQuantConfig,
)
from sc_evae.models.common import (
    GeneCountEmbedding,
    extract_pert_cond_input,
    make_pert_cond_module,
)
from sc_evae.models.masked_diffusion import SinusoidalTimeEmbedding
from sc_evae.models.transformer import build_backbone


class FlowMatchingModel(nn.Module):
    """
    Conditional flow matching prior.

    Parameters
    ----------
    config : FlowMatchingModelConfig
    vae    : optional frozen sc_evae with NoQuantConfig quantizer (Mode A).
    """

    def __init__(
        self,
        config: FlowMatchingModelConfig,
        vae=None,  # sc_evae | None
    ):
        super().__init__()
        self.config = config
        d_model = config.transformer.d_model

        if vae is not None:
            # ------------------------------------------------------------------
            # Mode A: flow over continuous VAE latent codes
            # ------------------------------------------------------------------
            if not isinstance(vae.config.quantizer, NoQuantConfig):
                raise ValueError(
                    "FlowMatchingModel (Mode A) requires a NoQuantConfig VAE — "
                    "the flow runs in the continuous latent space.  Got "
                    f"{type(vae.config.quantizer).__name__}."
                )
            for p in vae.parameters():
                p.requires_grad_(False)
            vae.eval()
            self.vae = vae

            self.d_latent: int = vae.config.quantizer.latent_dim
            self.num_latents: int = vae.config.num_latent_tokens
            self.input_proj = nn.Linear(self.d_latent, d_model)
            self.pos_embed = nn.Embedding(self.num_latents, d_model)
            nn.init.normal_(self.pos_embed.weight, std=0.02)
            self.output_proj = nn.Linear(d_model, self.d_latent)
            self._mode = "vae"
        else:
            # ------------------------------------------------------------------
            # Mode B: flow over log1p-normalized gene counts
            # ------------------------------------------------------------------
            self.vae = None
            self.num_genes: int = config.num_genes
            self.gene_tokens = nn.Embedding(config.num_genes, d_model)
            nn.init.normal_(self.gene_tokens.weight, std=0.02)
            # Project the (noisy) scalar per gene and add to gene embeddings.
            self.count_proj_in = GeneCountEmbedding(config.input_head, d_model)
            self.output_proj = nn.Linear(d_model, 1)
            self._mode = "counts"

        # Time embedding (always on — flow needs it).
        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

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

        self.transformer = build_backbone(config.transformer)

    def train(self, mode: bool = True):
        """Keep the frozen VAE in eval mode even when this model is set to train().

        nn.Module.train() recurses into all submodules, so without this override
        the frozen self.vae (a submodule) would be flipped back to training mode
        every epoch, re-enabling its dropout/batchnorm-in-train-mode behavior.
        """
        super().train(mode)
        if self.vae is not None:
            self.vae.eval()
        return self

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    def _build_cond(
        self,
        t: torch.Tensor,
        pert_emb: torch.Tensor | None,
        cls_cond: torch.Tensor | None = None,
        cell_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond = self.time_mlp(self.time_embed(t))  # (B, d_model)
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

    def _embed_inputs(self, x_t: torch.Tensor) -> torch.Tensor:
        """Map x_t to (B, L, d_model) — L = num_latents (A) or num_genes (B)."""
        if self._mode == "vae":
            B, L, _ = x_t.shape
            h = self.input_proj(x_t)
            pos = torch.arange(L, device=x_t.device)
            return h + self.pos_embed(pos).unsqueeze(0)
        # counts mode: x_t is (B, G)
        B, G = x_t.shape
        gene_ids = torch.arange(G, device=x_t.device)
        gene_embeds = self.gene_tokens(gene_ids).unsqueeze(0).expand(B, -1, -1)
        return self.count_proj_in(x_t, gene_embeds)

    def _project_velocity(self, h: torch.Tensor) -> torch.Tensor:
        """Map (B, L, d_model) → velocity at the input resolution."""
        if self._mode == "vae":
            return self.output_proj(h)  # (B, L, d_z)
        return self.output_proj(h).squeeze(-1)  # (B, G)

    def _velocity_field(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        pert_emb: torch.Tensor | None,
        cls_cond: torch.Tensor | None = None,
        cell_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Model's predicted velocity at (x_t, t).  Same shape as x_t."""
        h = self._embed_inputs(x_t)
        cond = self._build_cond(t, pert_emb, cls_cond, cell_type_idx=cell_type_idx)
        h = self.transformer(h, cond)  # DiT adaLN
        return self._project_velocity(h)

    # --------------------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------------------

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        pert_emb = extract_pert_cond_input(batch, self.config)
        cell_type_idx = (
            batch.get("cell_type_idx") if self.config.do_celltype_cond else None
        )
        cls_cond = batch.get("cls_cond")

        if self.vae is not None:
            with torch.no_grad():
                # Encode counts → pre-quant latent → Quantizer for mu/KL.
                pre_quant = self.vae.encode(batch["counts"])  # (B, L, d_z)
                q_out = self.vae.quantizer(pre_quant)
                x1 = q_out.codes  # (B, L, d_z) continuous
        else:
            x1 = batch["counts"]  # (B, G) log1p

        B = x1.shape[0]
        device = x1.device

        # 1. Sample flow time t ~ U(0, 1), one per sample.
        t = torch.rand(B, device=device)
        # 2. Source noise x0 ~ N(0, I), same shape as x1.
        x0 = torch.randn_like(x1)
        # 3. Straight-line interpolation.
        t_shape = (B,) + (1,) * (x1.ndim - 1)
        t_view = t.view(t_shape)
        x_t = (1.0 - t_view) * x0 + t_view * x1
        # 4. Target velocity is constant along the line.
        target_v = x1 - x0

        pred_v = self._velocity_field(
            x_t, t, pert_emb, cls_cond, cell_type_idx=cell_type_idx
        )
        loss = F.mse_loss(pred_v, target_v)

        with torch.no_grad():
            metrics = {
                "t_mean": t.mean().detach(),
                "v_target_rms": target_v.pow(2).mean().sqrt().detach(),
                "v_pred_rms": pred_v.pow(2).mean().sqrt().detach(),
            }
        return loss, metrics

    # --------------------------------------------------------------------------
    # Sampling / prediction
    # --------------------------------------------------------------------------

    @torch.no_grad()
    def _euler_sample(
        self,
        batch_size: int,
        shape_tail: tuple[int, ...],
        num_steps: int,
        pert_emb: torch.Tensor | None,
        device: torch.device,
        cls_cond: torch.Tensor | None = None,
        cell_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Euler integration of dx/dt = v_theta from t=0 to t=1."""
        x = torch.randn((batch_size,) + shape_tail, device=device)
        dt = 1.0 / max(num_steps, 1)
        for k in range(num_steps):
            t = torch.full((batch_size,), k * dt, device=device)
            v = self._velocity_field(
                x, t, pert_emb, cls_cond, cell_type_idx=cell_type_idx
            )
            x = x + dt * v
        return x

    @torch.no_grad()
    def _heun_sample(
        self,
        batch_size: int,
        shape_tail: tuple[int, ...],
        num_steps: int,
        pert_emb: torch.Tensor | None,
        device: torch.device,
        cls_cond: torch.Tensor | None = None,
        cell_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Heun predictor-corrector (2nd-order RK) integration from t=0 to t=1.

        Costs 2 NFE per step but achieves 2nd-order accuracy — 4 Heun steps
        (8 NFE) typically outperforms Euler with 16+ steps on straight-line flows.
        """
        x = torch.randn((batch_size,) + shape_tail, device=device)
        dt = 1.0 / max(num_steps, 1)
        for k in range(num_steps):
            t_cur = torch.full((batch_size,), k * dt, device=device)
            t_nxt = torch.full((batch_size,), (k + 1) * dt, device=device)
            v1 = self._velocity_field(
                x, t_cur, pert_emb, cls_cond, cell_type_idx=cell_type_idx
            )
            x_pred = x + dt * v1
            v2 = self._velocity_field(
                x_pred, t_nxt, pert_emb, cls_cond, cell_type_idx=cell_type_idx
            )
            x = x + 0.5 * dt * (v1 + v2)
        return x

    def _pseudo_sample_vae_mode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Mode A proxy: 4-step Heun integration from Gaussian noise in latent space,
        then decode through the VAE using the observed library size.
        """
        counts = batch["counts"]
        pert_emb = extract_pert_cond_input(batch, self.config)
        cell_type_idx = (
            batch.get("cell_type_idx") if self.config.do_celltype_cond else None
        )
        cls_cond = batch.get("cls_cond")
        B = counts.shape[0]
        sampled = self._heun_sample(
            B,
            (self.num_latents, self.d_latent),
            4,
            pert_emb,
            counts.device,
            cls_cond=cls_cond,
            cell_type_idx=cell_type_idx,
        )  # (B, L, d_z) — continuous latents matching the VAE's NoQuant output space
        pert_emb_dec = extract_pert_cond_input(batch, self.vae.config)
        cell_type_idx_dec = (
            batch.get("cell_type_idx") if self.vae.config.do_celltype_cond else None
        )
        return self.vae.decode_observed(
            sampled,
            pert_emb_dec,
            counts.sum(-1, keepdim=True),
            cell_type_idx=cell_type_idx_dec,
        )

    @torch.no_grad()
    def pseudo_sample(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Return predicted mean expression (B, G) for proxy validation metrics.

        Mode A: 4-step Heun integration in latent space → VAE decode.
        Mode B: 4-step Heun integration directly in count space.
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
        return self._heun_sample(
            B,
            (G,),
            4,
            pert_emb,
            counts.device,
            cls_cond=cls_cond,
            cell_type_idx=cell_type_idx,
        )

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
        Generate `batch_size` samples via Heun (RK2) integration.

        Returns:
          Mode A: (B, L, d_z) continuous latent codes.
          Mode B: (B, G) log1p-normalised counts.
        """
        num_steps = num_steps or self.config.num_sample_steps
        device = torch.device(device)
        if self._mode == "vae":
            shape_tail = (self.num_latents, self.d_latent)
        else:
            shape_tail = (self.num_genes,)
        return self._heun_sample(
            batch_size,
            shape_tail,
            num_steps,
            pert_emb,
            device,
            cls_cond=cls_cond,
            cell_type_idx=cell_type_idx,
        )
