"""sc_evae (FSQ-VQ-VAE) model config."""

from dataclasses import dataclass, field

from sc_evae.config.models.base import ModelConfig
from sc_evae.config.models.heads import (
    InputHeadConfig,
    MLPInputHeadConfig,
    MSEHeadConfig,
    OutputHeadConfig,
    _CONTINUOUS_HEAD_TYPES,
    _DISCRETE_HEAD_TYPES,
    _propagate_count_max,
    _propagate_is_count_discrete,
    _validate_input_head,
)
from sc_evae.config.models.quantizer import FSQConfig, QuantizerConfig
from sc_evae.config.models.transformer import TransformerConfig


@ModelConfig.register_subclass("vae")
@dataclass
class sc_evaeConfig(ModelConfig):
    num_latent_tokens: int = 64
    quantizer: QuantizerConfig = field(default_factory=FSQConfig)
    encoder: TransformerConfig = field(default_factory=TransformerConfig)
    decoder: TransformerConfig = field(default_factory=TransformerConfig)
    # Perturbation conditioning
    do_pert_cond: bool = False
    # Cell-type conditioning (additive adaLN along the pert path)
    do_celltype_cond: bool = False
    # Output head — each subclass carries only its own parameters
    output_head: OutputHeadConfig = field(default_factory=MSEHeadConfig)
    # Input head — maps each scalar count to embed_dim features.
    input_head: InputHeadConfig = field(default_factory=MLPInputHeadConfig)

    # scANVI-style perturbation classifier head (Xu et al. 2021) applied on
    # the mean-pooled pre-quantization encoder output (d_model-dim vector per
    # cell — the same representation eval_pfizer.py uses with
    # ``use_pre_quantproj=True``). The encoder does NOT receive pert labels;
    # the classifier reads them out, which is the pattern that empirically
    # gave better downstream metrics than encoder conditioning.
    #
    # ``num_perts`` is resolved at runtime from the dataset and does not need
    # to be set in YAML. Must be > 0 when train_pert_classifier=True.
    train_pert_classifier: bool = False
    # Multiplier on CE(clf(pooled_z), pert_idx) added to the total loss.
    # Reconstruction loss dominates by default; values in [1, 100] are
    # reasonable starting points.
    classifier_weight: float = 1.0
    # Hidden widths of the classifier MLP. Empty list = single Linear layer
    # (d_model → num_perts).
    classifier_hidden: list[int] = field(default_factory=list)
    num_perts: int | None = None

    # Perturbation-feature regression head (e.g. to ESM2 protein embeddings).
    # When True, adds a regressor head f(pooled_z) → R^pert_emb_dim trained
    # with cosine loss against batch["pert_emb"]. Requires
    # dataset.pert_mapping_path and pert_emb_dim > 0 (runtime-resolved).
    # Cells with zero pert_emb (perts missing from pert_mapping — e.g.
    # SAFE_TARGET, NO-TARGET) are masked out of the regression loss.
    train_pert_regressor: bool = False
    regressor_weight: float = 1.0
    regressor_hidden: list[int] = field(default_factory=list)

    def __post_init__(self):
        _CROSS_ATTN_TYPES = ("cross_attention", "dit_cross_attention")
        if self.encoder.type not in _CROSS_ATTN_TYPES:
            raise ValueError(
                f"VAE encoder must use cross-attention to attend over gene embeddings. "
                f"Got encoder.type={self.encoder.type!r}. "
                f"Use one of {_CROSS_ATTN_TYPES}."
            )
        if self.decoder.type not in _CROSS_ATTN_TYPES:
            raise ValueError(
                f"VAE decoder must use cross-attention to attend over code embeddings. "
                f"Got decoder.type={self.decoder.type!r}. "
                f"Use one of {_CROSS_ATTN_TYPES}."
            )

    def validate(self) -> None:
        super().validate()
        _validate_input_head(
            self.input_head, self.is_count_discrete, "sc_evaeConfig"
        )
        if (
            isinstance(self.output_head, _DISCRETE_HEAD_TYPES)
            and not self.is_count_discrete
        ):
            raise ValueError(
                f"{type(self.output_head).__name__} requires discrete count data "
                f"(is_count_discrete=True), but is_count_discrete=False. "
                f"Use MSEHeadConfig or HurdleHeadConfig for log1p-normalized data."
            )
        if (
            isinstance(self.output_head, _CONTINUOUS_HEAD_TYPES)
            and self.is_count_discrete
        ):
            raise ValueError(
                f"{type(self.output_head).__name__} requires continuous data "
                f"(is_count_discrete=False), but is_count_discrete=True. "
                f"Use NBHeadConfig or CrossEntropyHeadConfig for raw counts."
            )
        if self.train_pert_classifier:
            if self.num_perts is None or self.num_perts <= 0:
                raise ValueError(
                    f"sc_evaeConfig.train_pert_classifier=True requires "
                    f"num_perts > 0; got num_perts={self.num_perts}. This is "
                    f"normally resolved from ds.num_perts at runtime."
                )
        if self.train_pert_regressor:
            if self.pert_emb_dim is None or self.pert_emb_dim <= 0:
                raise ValueError(
                    f"sc_evaeConfig.train_pert_regressor=True requires "
                    f"pert_emb_dim > 0; got pert_emb_dim={self.pert_emb_dim}. Set "
                    f"dataset.pert_mapping_path in your YAML."
                )

    def set_attributes_from_dataset(self, **kwargs) -> None:
        super().set_attributes_from_dataset(**kwargs)
        _propagate_count_max(kwargs.get("count_max"), self.output_head, self.input_head)
        _propagate_is_count_discrete(self.is_count_discrete, self.input_head)
        if kwargs.get("num_perts") is not None:
            self.num_perts = int(kwargs["num_perts"])
