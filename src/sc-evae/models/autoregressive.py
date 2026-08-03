"""
Autoregressive prior model for sc_evae.

Two operating modes:

  Mode A — VAE latent codes (vae_path set in config):
    A frozen sc_evae encodes each cell into (B, L) discrete codebook indices.
    The AR model learns p(z_1, ..., z_L) using a causal transformer + cross-entropy head.
    forward() encodes under no_grad, shifts indices right, runs transformer, computes loss.
    pseudo_sample() runs a teacher-forced pass and decodes soft codes through the VAE
    for proxy perturbation metrics (a cheap stand-in for the real sample() below).
    sample() returns raw (B, L) int64 codebook indices — callers decode through the
    VAE externally (matches MaskedDiffusionModel / FlowMatchingModel sample contracts).

  Mode B — raw gene counts (vae_path is None):
    The AR model trains directly on raw integer counts over genes.
    Gene identity embeddings + sinusoidal count embeddings form the input sequence.
    forward() shifts counts right (BOS = 0), runs transformer, computes cross-entropy.
    pseudo_sample() returns teacher-forced expected counts via output_head.predict().
"""

import torch
import torch.nn as nn

from sc_evae.config.models import (
    AutoregressiveModelConfig,
    CrossEntropyHeadConfig,
    NoQuantConfig,
)
from sc_evae.models.common import (
    GeneCountEmbedding,
    extract_pert_cond_input,
    make_pert_cond_module,
)
from sc_evae.models.output_heads import make_output_head
from sc_evae.models.transformer import build_backbone


def _override_max_count(head_config, max_count: int):
    """Return a new head config with max_count replaced, preserving other fields."""
    if isinstance(head_config, CrossEntropyHeadConfig):
        return CrossEntropyHeadConfig(max_count=max_count)
    raise TypeError(
        f"_override_max_count: unsupported head config type {type(head_config).__name__}"
    )


class AutoregressiveModel(nn.Module):
    def __init__(
        self,
        config: AutoregressiveModelConfig,
        vae=None,  # sc_evae | None  (avoid circular import with type annotation)
    ):
        super().__init__()
        self.config = config
        d_model = config.transformer.d_model

        # Validate transformer type
        if config.transformer.type != "dit":
            raise ValueError(
                "AutoregressiveModel requires transformer.type='dit' so perturbation "
                f"conditioning can be applied via adaLN. Got {config.transformer.type!r}."
            )

        if vae is not None:
            # ------------------------------------------------------------------
            # Mode A: AR over VAE discrete latent codes
            # ------------------------------------------------------------------
            if isinstance(vae.config.quantizer, NoQuantConfig):
                raise ValueError(
                    "AutoregressiveModel requires a discrete VAE quantizer "
                    "(FSQConfig, VQConfig, or LFQConfig). NoQuantConfig produces "
                    "continuous latents that cannot be modelled autoregressively."
                )
            vocab_size: int = vae.config.quantizer.codebook_size

            # Freeze all VAE parameters — it is a sub-module so it moves to device
            # with the AR model and is saved in checkpoints for self-containedness.
            for p in vae.parameters():
                p.requires_grad_(False)
            vae.eval()
            self.vae = vae

            # BOS token id = vocab_size (one beyond the valid code range [0, vocab_size-1])
            self.bos_token_id: int = vocab_size
            self.token_embed = nn.Embedding(vocab_size + 1, d_model)
            nn.init.normal_(self.token_embed.weight, std=0.02)

            # Auto-configure the output head to match the exact codebook vocabulary.
            # The user controls the head type (cross_entropy); max_count
            # is always overridden here so the user never has to look up the codebook size.
            head_config = _override_max_count(config.output_head, vocab_size - 1)
        else:
            # ------------------------------------------------------------------
            # Mode B: AR over raw gene counts
            # ------------------------------------------------------------------
            self.vae = None
            # Gene identity embeddings serve as positional/content tokens.
            self.gene_tokens = nn.Embedding(config.num_genes, d_model)
            nn.init.normal_(self.gene_tokens.weight, std=0.02)
            # Count embedding: adds count features to gene embeddings.
            self.count_proj_in = GeneCountEmbedding(config.input_head, d_model)
            head_config = config.output_head

        self.transformer = build_backbone(config.transformer)
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
        self.output_head = make_output_head(
            head_config, d_model, num_genes=config.num_genes
        )

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
    # Internal helpers
    # --------------------------------------------------------------------------

    def _build_cond(
        self,
        pert_emb: torch.Tensor | None,
        cls_cond: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        cell_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Zero-init adaLN cond + pert_emb projection + cell-type embedding + optional CLS."""
        d_model = self.config.transformer.d_model
        cond = torch.zeros(batch_size, d_model, device=device)
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

    def _cond_from_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        counts = batch["counts"]
        pert_emb = extract_pert_cond_input(batch, self.config)
        cell_type_idx = (
            batch.get("cell_type_idx") if self.config.do_celltype_cond else None
        )
        return self._build_cond(
            pert_emb=pert_emb,
            cls_cond=batch.get("cls_cond"),
            batch_size=counts.shape[0],
            device=counts.device,
            cell_type_idx=cell_type_idx,
        )

    def _forward_vae_mode(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict]:
        counts = batch["counts"]
        cond = self._cond_from_batch(batch)

        with torch.no_grad():
            pre_quant = self.vae.encode(counts)
            q_out = self.vae.quantizer(pre_quant)
            targets = q_out.indices  # (B, L) int64

        B, L = targets.shape
        bos = targets.new_full((B, 1), self.bos_token_id)
        input_ids = torch.cat([bos, targets[:, :-1]], dim=1)  # (B, L) shifted right
        x = self.token_embed(input_ids)  # (B, L, d_model)
        x = self.transformer(x, cond)  # (B, L, d_model)
        loss, aux = self.output_head(x, targets.float())
        return loss, aux

    def _forward_gene_mode(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict]:
        counts = batch["counts"]  # (B, G) raw integer counts
        cond = self._cond_from_batch(batch)
        B, G = counts.shape
        gene_ids = torch.arange(G, device=counts.device)
        gene_embeds = self.gene_tokens(gene_ids).unsqueeze(0).expand(B, -1, -1)

        # Shift right: prepend BOS count (0), drop last gene count.
        shifted = torch.cat([counts.new_zeros(B, 1), counts[:, :-1]], dim=1)
        x = self.count_proj_in(shifted.float(), gene_embeds)  # (B, G, d_model)
        x = self.transformer(x, cond)  # (B, G, d_model)
        loss, aux = self.output_head(x, counts.float())
        return loss, aux

    # --------------------------------------------------------------------------
    # Public API — matches sc_evae interface
    # --------------------------------------------------------------------------

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """
        Compute AR cross-entropy loss.

        Returns
        -------
        loss   : scalar tensor
        metrics: dict of auxiliary scalars from the output head
        """
        if self.vae is not None:
            return self._forward_vae_mode(batch)
        return self._forward_gene_mode(batch)

    def _pseudo_sample_vae_mode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Mode A proxy: teacher-forced pass over ground-truth code indices, then
        decode through the VAE using soft codes (expected code vector under the
        predicted distribution).  Uses observed library size from the batch.
        """
        counts = batch["counts"]
        cond = self._cond_from_batch(batch)

        pre_quant = self.vae.encode(counts)
        q_out = self.vae.quantizer(pre_quant)
        targets = q_out.indices  # (B, L)

        B, L = targets.shape
        bos = targets.new_full((B, 1), self.bos_token_id)
        input_ids = torch.cat([bos, targets[:, :-1]], dim=1)
        x = self.token_embed(input_ids)
        x = self.transformer(x, cond)  # (B, L, d_model)

        # Soft codes: expected code vector under the model's predicted distribution.
        logits = self.output_head.proj(x)  # (B, L, vocab_size)
        probs = logits.softmax(dim=-1)  # (B, L, vocab_size)
        codebook = self.vae.quantizer.all_codes(counts.device)  # (vocab_size, d_z)
        soft_codes = probs @ codebook  # (B, L, d_z)

        pert_emb = extract_pert_cond_input(batch, self.vae.config)
        cell_type_idx = (
            batch.get("cell_type_idx") if self.vae.config.do_celltype_cond else None
        )
        return self.vae.decode_observed(
            soft_codes,
            pert_emb,
            counts.sum(-1, keepdim=True),
            cell_type_idx=cell_type_idx,
        )

    @torch.no_grad()
    def pseudo_sample(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Return predicted mean expression (B, G) for proxy validation metrics.

        Mode A (VAE): teacher-forced AR pass → soft codes → VAE decode.
        Mode B (gene AR): teacher-forced expected-count via output_head.predict().
        """
        if self.vae is not None:
            return self._pseudo_sample_vae_mode(batch)

        counts = batch["counts"]
        cond = self._cond_from_batch(batch)
        B, G = counts.shape
        gene_ids = torch.arange(G, device=counts.device)
        gene_embeds = self.gene_tokens(gene_ids).unsqueeze(0).expand(B, -1, -1)
        shifted = torch.cat([counts.new_zeros(B, 1), counts[:, :-1]], dim=1)
        x = self.count_proj_in(shifted.float(), gene_embeds)
        x = self.transformer(x, cond)
        return self.output_head.predict(x)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        pert_emb: torch.Tensor | None = None,
        temperature: float = 1.0,
        device: torch.device | str = "cpu",
        cls_cond: torch.Tensor | None = None,
        cell_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate ``batch_size`` cells autoregressively.

        Parameters
        ----------
        batch_size    : number of cells to generate.
        pert_emb      : perturbation embeddings ``(batch_size, pert_emb_dim)``.
                        Required iff ``config.do_pert_cond=True``.
        cell_type_idx : int64 cell-type indices ``(batch_size,)``.
                        Required iff ``config.do_celltype_cond=True``.
        temperature   : softmax temperature.  1.0 = categorical; lower → greedier;
                        higher → more uniform.
        device        : generation device.

        Returns
        -------
        Mode A: ``(B, L)`` int64 VAE codebook indices.  Caller decodes through
                the frozen VAE (matches MDLM / flow_matching sample contracts).
        Mode B: ``(B, G)`` sampled integer counts as a float tensor.
                Note: Mode B requires O(G) transformer passes (one per gene) and
                is slow for large gene vocabularies.
        """
        device = torch.device(device)
        pert_emb_dev = pert_emb.to(device) if pert_emb is not None else None
        cls_cond_dev = cls_cond.to(device) if cls_cond is not None else None
        cell_type_idx_dev = (
            cell_type_idx.to(device) if cell_type_idx is not None else None
        )
        cond = self._build_cond(
            pert_emb=pert_emb_dev,
            cls_cond=cls_cond_dev,
            batch_size=batch_size,
            device=device,
            cell_type_idx=cell_type_idx_dev,
        )

        if self.vae is not None:
            return self._sample_vae_mode(batch_size, cond, temperature, device)
        return self._sample_gene_mode(batch_size, cond, temperature, device)

    def _sample_vae_mode(
        self,
        batch_size: int,
        cond: torch.Tensor,
        temperature: float,
        device: torch.device,
    ) -> torch.Tensor:
        L = self.vae.config.num_latent_tokens
        # Growing sequence starting with the BOS token.
        tokens = torch.full(
            (batch_size, 1), self.bos_token_id, dtype=torch.long, device=device
        )
        for _ in range(L):
            x = self.token_embed(tokens)  # (B, l+1, d_model)
            x = self.transformer(x, cond)  # (B, l+1, d_model) — causal mask applied
            logits = self.output_head.proj(x[:, -1])  # (B, vocab_size)
            if temperature != 1.0:
                logits = logits / temperature
            next_token = torch.multinomial(
                logits.softmax(dim=-1), num_samples=1
            )  # (B, 1)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens[:, 1:]  # (B, L) int64 — strip BOS; caller decodes through VAE

    def _sample_gene_mode(
        self,
        batch_size: int,
        cond: torch.Tensor,
        temperature: float,
        device: torch.device,
    ) -> torch.Tensor:
        G = self.config.num_genes
        gene_ids = torch.arange(G, device=device)

        # counts_seq holds the shifted input: [BOS=0, c_0, c_1, ..., c_{g-1}].
        counts_seq = torch.zeros(batch_size, 1, device=device)
        sampled_counts: list[torch.Tensor] = []

        for g in range(G):
            gene_embeds = (
                self.gene_tokens(gene_ids[: g + 1])
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )  # (B, g+1, d_model)
            x = self.count_proj_in(counts_seq, gene_embeds)  # (B, g+1, d_model)
            x = self.transformer(x, cond)  # (B, g+1, d_model)
            logits = self.output_head.proj(x[:, -1])  # (B, max_count+1)
            if temperature != 1.0:
                logits = logits / temperature
            next_count = torch.multinomial(
                logits.softmax(dim=-1), num_samples=1
            ).float()  # (B, 1)
            sampled_counts.append(next_count)
            counts_seq = torch.cat([counts_seq, next_count], dim=1)

        return torch.cat(sampled_counts, dim=1)  # (B, G)
