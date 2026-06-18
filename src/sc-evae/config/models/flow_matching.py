"""Flow-matching prior model config (continuous)."""

from dataclasses import dataclass, field

from sc_evae.config.models.base import ModelConfig
from sc_evae.config.models.heads import (
    InputHeadConfig,
    MLPInputHeadConfig,
    _propagate_count_max,
    _propagate_is_count_discrete,
    _validate_input_head,
)
from sc_evae.config.models.transformer import TransformerConfig


@ModelConfig.register_subclass("flow_matching")
@dataclass
class FlowMatchingModelConfig(ModelConfig):
    """
    Conditional flow matching prior (Lipman et al. / Rectified Flow variant).

    Straight-line interpolation between Gaussian noise ``x0 ~ N(0, I)`` and
    clean data ``x1`` with constant target velocity ``v = x1 - x0``.

    Two modes:
      Mode A: ``vae_path`` set → flow in the VAE's continuous latent space
              (requires a NoQuantConfig VAE).
      Mode B: ``vae_path=None`` → flow over gene counts directly (requires
              is_count_discrete=False, i.e. log1p-normalized).
    """

    vae_path: str | None = None

    transformer: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(type="dit", causal_input_mask=False)
    )

    num_sample_steps: int = 128
    do_pert_cond: bool = True
    # When True, look up batch["cell_type_idx"] in an nn.Embedding and add
    # into the adaLN cond vector (same additive path as pert).
    do_celltype_cond: bool = False

    # --- Mode B (raw gene counts) input embedding ---
    # Flow matching is continuous, so EmbeddingInputHeadConfig is rejected at
    # validate() time against is_count_discrete=False.
    input_head: InputHeadConfig = field(default_factory=MLPInputHeadConfig)

    def __post_init__(self):
        if self.transformer.causal_input_mask:
            raise ValueError(
                "FlowMatchingModelConfig requires a bidirectional transformer "
                "(causal_input_mask=False)."
            )
        if "cross_attention" in self.transformer.type:
            raise ValueError(
                f"FlowMatchingModelConfig does not support cross-attention "
                f"transformer types (got {self.transformer.type!r}). "
                f"Use 'dit'."
            )
        if self.transformer.type != "dit":
            raise ValueError(
                f"FlowMatchingModelConfig requires transformer.type='dit' "
                f"(adaLN modulation for flow time).  Got {self.transformer.type!r}."
            )

    def validate(self) -> None:
        if self.vae_path is None and self.is_count_discrete is True:
            raise ValueError(
                "FlowMatchingModelConfig with vae_path=None trains directly on gene "
                "counts and requires continuous data (is_count_discrete=False). "
                "Set dataset.apply_log1p=True in your config."
            )
        if self.vae_path is None:
            _validate_input_head(
                self.input_head, self.is_count_discrete, "FlowMatchingModelConfig"
            )

    def set_attributes_from_dataset(self, **kwargs) -> None:
        super().set_attributes_from_dataset(**kwargs)
        _propagate_count_max(kwargs.get("count_max"), self.input_head)
        _propagate_is_count_discrete(self.is_count_discrete, self.input_head)
