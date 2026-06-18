"""Masked (absorbing) discrete diffusion prior config (MDLM / RADD)."""

from dataclasses import dataclass, field

from sc_evae.config.models.base import ModelConfig
from sc_evae.config.models.heads import (
    CrossEntropyHeadConfig,
    EmbeddingInputHeadConfig,
    InputHeadConfig,
    OutputHeadConfig,
    _propagate_count_max,
    _propagate_is_count_discrete,
    _validate_input_head,
)
from sc_evae.config.models.transformer import TransformerConfig

_MDLM_HEAD_TYPES = (CrossEntropyHeadConfig,)


@ModelConfig.register_subclass("masked_diffusion")
@dataclass
class MaskedDiffusionModelConfig(ModelConfig):
    """
    Masked (absorbing) discrete diffusion prior.

    Two modes:
      Mode A: ``vae_path`` set → model learns p(z_1..z_L) over the VAE's
              discrete latent codes (requires an FSQ/VQ/LFQ VAE).
      Mode B: ``vae_path=None`` → model trains directly on raw integer gene
              counts (requires is_count_discrete=True).

    Two variants (both absorbing / BERT-style objective; see
    docs/planning/absorbing_diffussion.md):

      * MDLM   (time_conditioned=True,  default) — the denoiser sees t via
                adaLN modulation.  Matches Sahoo et al. 2024.
      * RADD   (time_conditioned=False)          — t is dropped from the
                network entirely (Ou et al. 2024).

    The loss is the same in both cases: `(1/lambda) * CE` on masked positions
    with ``lambda = t`` under the log-linear schedule ``alpha_t = 1 - t``.
    """

    vae_path: str | None = None

    # Bidirectional transformer.  adaLN-Zero DiT is used by default so time
    # (and optionally perturbation) conditioning flows through adaLN.
    transformer: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(type="dit", causal_input_mask=False)
    )

    # max_count is auto-overridden at model construction time: codebook_size-1
    # in Mode A, data-derived count_max in Mode B.
    output_head: OutputHeadConfig = field(default_factory=CrossEntropyHeadConfig)

    # --- Mode B (raw counts) input embedding ---
    # Maps each scalar count to d_model features.  Defaults to a learned
    # lookup table; sinusoidal is valid for discrete counts; MLP is rejected
    # by ``_validate_input_head`` because Mode B requires is_count_discrete=True.
    input_head: InputHeadConfig = field(default_factory=EmbeddingInputHeadConfig)

    # --- Diffusion objective ---
    # MDLM = True, RADD = False.
    time_conditioned: bool = True
    # Multiplies t before sinusoidal time embedding. Useful when t is sampled
    # in [0, 1], where scale=1 can make many channels near-constant.
    time_embedding_scale: float = 1000.0
    # Base period for sinusoidal time embedding frequencies.
    time_embedding_max_period: float = 10000.0
    # Numerical stability: clamp 1/lambda weight so a single rare mask at
    # small t does not blow up the loss.  Also used as low/high-t truncation.
    time_epsilon: float = 1e-3
    num_sample_steps: int = 128

    do_pert_cond: bool = True
    # When True, look up batch["cell_type_idx"] in an nn.Embedding and add
    # into the adaLN cond vector (same additive path as pert).
    do_celltype_cond: bool = False

    def __post_init__(self):
        if not isinstance(self.output_head, _MDLM_HEAD_TYPES):
            raise ValueError(
                f"MaskedDiffusionModelConfig requires a cross_entropy output head, "
                f"got {type(self.output_head).__name__}."
            )
        if self.transformer.causal_input_mask:
            raise ValueError(
                "MaskedDiffusionModelConfig requires a bidirectional transformer "
                "(causal_input_mask=False).  The absorbing-diffusion objective is "
                "BERT-style and must see all unmasked context."
            )
        if "cross_attention" in self.transformer.type:
            raise ValueError(
                f"MaskedDiffusionModelConfig does not support cross-attention "
                f"transformer types (got {self.transformer.type!r}). "
                f"Use 'plain' or 'dit'."
            )
        needs_cond = self.time_conditioned or self.do_pert_cond or self.do_celltype_cond
        if needs_cond and self.transformer.type != "dit":
            raise ValueError(
                f"time_conditioned, do_pert_cond, or do_celltype_cond requires "
                f"transformer.type='dit' (adaLN modulation).  "
                f"Got {self.transformer.type!r}."
            )

    def validate(self) -> None:
        if self.vae_path is None and self.is_count_discrete is False:
            raise ValueError(
                "MaskedDiffusionModelConfig with vae_path=None trains directly on raw "
                "counts and requires discrete count data (is_count_discrete=True). "
                "Set dataset.apply_log1p=False in your config."
            )
        if self.vae_path is None:
            _validate_input_head(
                self.input_head, self.is_count_discrete, "MaskedDiffusionModelConfig"
            )

    def set_attributes_from_dataset(self, **kwargs) -> None:
        super().set_attributes_from_dataset(**kwargs)
        _propagate_count_max(kwargs.get("count_max"), self.output_head, self.input_head)
        _propagate_is_count_discrete(self.is_count_discrete, self.input_head)
