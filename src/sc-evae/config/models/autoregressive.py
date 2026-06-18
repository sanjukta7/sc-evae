"""Autoregressive prior model config."""

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

_AR_HEAD_TYPES = (CrossEntropyHeadConfig,)


@ModelConfig.register_subclass("autoregressive")
@dataclass
class AutoregressiveModelConfig(ModelConfig):
    # Path to a VAE training run directory (contains config.yaml + checkpoints/).
    # When set: the VAE is loaded frozen and the AR model learns p(z_1, ..., z_L)
    # over the discrete latent codes. When None: AR is trained directly on raw
    # integer gene counts (requires is_count_discrete=True).
    vae_path: str | None = None

    # Causal transformer backbone (DiT by default).
    transformer: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(type="dit", causal_input_mask=True)
    )

    # Output head — must be a cross-entropy head.
    # When vae_path is set, output_head.max_count is auto-overridden to
    # match the VAE's codebook_size - 1 at model construction time.
    output_head: OutputHeadConfig = field(default_factory=CrossEntropyHeadConfig)

    # Only used when vae_path is None (AR on raw integer counts).
    input_head: InputHeadConfig = field(default_factory=EmbeddingInputHeadConfig)

    # When True, project batch["pert_emb"] into the adaLN cond vector.
    # When False, the model is unconditional on perturbation (matches MDLM/Flow).
    do_pert_cond: bool = True
    # When True, look up batch["cell_type_idx"] in an nn.Embedding and add
    # into the adaLN cond vector (same additive path as pert).
    do_celltype_cond: bool = False

    def __post_init__(self):
        if not isinstance(self.output_head, _AR_HEAD_TYPES):
            raise ValueError(
                f"AutoregressiveModelConfig requires a cross_entropy output head, "
                f"got {type(self.output_head).__name__}. "
                f"NB, MSE, Hurdle heads do not apply to autoregressive models."
            )

    def validate(self) -> None:
        if self.vae_path is None and self.is_count_discrete is False:
            raise ValueError(
                "AutoregressiveModelConfig with vae_path=None trains directly on raw counts "
                "and requires discrete count data (is_count_discrete=True). "
                "Set dataset.apply_log1p=False in your config."
            )
        if self.vae_path is None:
            _validate_input_head(
                self.input_head, self.is_count_discrete, "AutoregressiveModelConfig"
            )

    def set_attributes_from_dataset(self, **kwargs) -> None:
        super().set_attributes_from_dataset(**kwargs)
        _propagate_count_max(kwargs.get("count_max"), self.output_head, self.input_head)
        _propagate_is_count_discrete(self.is_count_discrete, self.input_head)
