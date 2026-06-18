import torch
import torch.nn as nn

from sc_evae.config.models import (
    EmbeddingInputHeadConfig,
    InputHeadConfig,
    MLPInputHeadConfig,
    SinusoidalInputHeadConfig,
)


class _CountMLP(nn.Module):
    """2-layer MLP on a scalar count.  (B, L) → (B, L, D).

    Both linears are bias-free so that ``MLP(0) = 0``.  Combined with
    multiplicative combination in ``GeneCountEmbedding``, a zero count
    produces the zero vector — a natural null token for sparse counts.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        return self.mlp(counts.unsqueeze(-1).float())


class _CountSinusoidal(nn.Module):
    """Sinusoidal features of a scalar count followed by a learned linear projection.

    The underlying scheme is the standard transformer positional encoding with
    base 10000: frequencies span from 1 (fastest) to 1/10000 (slowest).  Pick
    ``scale`` so ``count * scale`` lands roughly in [0, 10000].
    """

    def __init__(self, embed_dim: int, scale: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = scale
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        D = self.embed_dim
        half_D = D // 2
        i = torch.arange(half_D, device=counts.device, dtype=torch.float32)
        div_term = torch.pow(10000.0, i / half_D)
        pos = (counts.float() * self.scale).unsqueeze(-1)
        enc = pos / div_term
        encoding = torch.cat([torch.sin(enc), torch.cos(enc)], dim=-1)[..., :D]
        return self.proj(encoding)


class _CountEmbedding(nn.Module):
    """Learned lookup table over integer counts, clamped at ``max_count``.

    Row 0 is zero-initialised so that count 0 (or bin 0 under quantile
    binning) yields the zero vector — a natural null token under
    multiplicative combination with gene embeddings.  Other rows keep a
    small random init so they are distinguishable at step 0.
    """

    def __init__(self, embed_dim: int, max_count: int):
        super().__init__()
        self.max_count = max_count
        self.embedding = nn.Embedding(max_count + 1, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        return self.embedding(counts.long().clamp(0, self.max_count))


class GeneCountEmbedding(nn.Module):
    """
    Unified input head: projects a scalar expression count to ``embed_dim``
    features and multiplicatively modulates gene-identity embeddings
    (scLDM-style ``gene_emb * f(count)``).  Zero-count genes collapse to
    the zero vector whenever ``f(0) = 0`` (MLP without bias; embedding
    row 0 zero-initialised), giving a natural null token over the ~90%
    zero entries of single-cell counts.

    The specific count projection is selected by the ``InputHeadConfig``
    subtype — dispatch is via ``isinstance``, never string matching.
    Any extra parameters (e.g. ``scale``, ``max_count``, ``is_count_discrete``)
    are pulled from the config so callers never have to thread them through.
    """

    def __init__(self, config: InputHeadConfig, embed_dim: int):
        super().__init__()
        if isinstance(config, MLPInputHeadConfig):
            self.count_proj = _CountMLP(embed_dim)
            # log1p inside the head when upstream counts are discrete (raw
            # integer counts).  Other variants handle range differently —
            # sinusoidal via ``scale``, embedding via integer lookup — so
            # they never log1p internally.
            self._apply_log1p = bool(config.is_count_discrete)
        elif isinstance(config, SinusoidalInputHeadConfig):
            self.count_proj = _CountSinusoidal(embed_dim, scale=config.scale)
            self._apply_log1p = False
        elif isinstance(config, EmbeddingInputHeadConfig):
            self.count_proj = _CountEmbedding(embed_dim, config.max_count)
            self._apply_log1p = False
        else:
            raise TypeError(f"Unknown InputHeadConfig type: {type(config).__name__}")

    def forward(
        self,
        counts: torch.Tensor,  # (B, L) scalar per gene
        gene_embeds: torch.Tensor,  # (B, L, D) gene-identity embeddings
    ) -> torch.Tensor:  # (B, L, D)
        if self._apply_log1p:
            counts = torch.log1p(counts.float())
        return gene_embeds * self.count_proj(counts)


# ---------------------------------------------------------------------------
# Perturbation conditioning helper
# ---------------------------------------------------------------------------
#
# Two ways to feed perturbation identity into a model's adaLN cond vector:
#
#   'projected' — project a precomputed ``batch['pert_emb']`` (e.g. ESM2 protein
#                 features) via ``nn.Linear(pert_emb_dim, d_model)``.
#   'learned'   — look up ``batch['pert_idx']`` in a learnable
#                 ``nn.Embedding(num_perts, d_model)``. No external features.
#
# Both produce a ``(B, d_model)`` tensor, so downstream conditioning code is
# unchanged.


def make_pert_cond_module(config, d_model: int) -> nn.Module | None:
    """Build the pert-conditioning module for a model with ``do_pert_cond=True``.

    Returns ``None`` when ``config.do_pert_cond`` is False (or absent), so the
    caller can wire ``self.pert_proj = make_pert_cond_module(config, d_model)``
    once and dispatch on ``self.pert_proj is not None`` thereafter.
    """
    if not getattr(config, "do_pert_cond", False):
        return None
    source = getattr(config, "pert_cond_source", "projected")
    if source == "learned":
        if not config.num_perts or config.num_perts <= 0:
            raise ValueError(
                "make_pert_cond_module: pert_cond_source='learned' requires "
                f"num_perts > 0; got num_perts={config.num_perts}."
            )
        return nn.Embedding(config.num_perts, d_model)
    if source == "projected":
        if not config.pert_emb_dim or config.pert_emb_dim <= 0:
            raise ValueError(
                "make_pert_cond_module: pert_cond_source='projected' requires "
                f"pert_emb_dim > 0; got pert_emb_dim={config.pert_emb_dim}. "
                "Set DatasetConfig.pert_mapping_path or use 'learned'."
            )
        return nn.Linear(config.pert_emb_dim, d_model)
    raise ValueError(
        f"make_pert_cond_module: unknown pert_cond_source={source!r} "
        "(expected 'projected' or 'learned')."
    )


def extract_pert_cond_input(
    batch: dict[str, torch.Tensor], config
) -> torch.Tensor | None:
    """Pull the right batch tensor for the configured pert-cond source.

    Returns ``batch['pert_idx']`` for 'learned' mode and ``batch['pert_emb']``
    for 'projected' mode, or ``None`` when the model has ``do_pert_cond=False``
    (or the relevant key is absent — the caller decides whether that is fatal).
    """
    if not getattr(config, "do_pert_cond", False):
        return None
    source = getattr(config, "pert_cond_source", "projected")
    if source == "learned":
        return batch.get("pert_idx")
    return batch.get("pert_emb")
