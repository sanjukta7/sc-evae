"""scVI baseline model config."""

from dataclasses import dataclass, field

from sc_evae.config.models.base import ModelConfig


@ModelConfig.register_subclass("scvi")
@dataclass
class ScVIConfig(ModelConfig):
    """
    scVI (Lopez et al. 2018) — MLP-based VAE with NB reconstruction and
    factored (observed) library size: ``μ_g = ℓ · softmax(decoder(z))_g``.

    Expects raw integer-valued counts — set
    ``dataset.apply_normalize=False``, ``dataset.apply_log1p=False``,
    ``dataset.use_binning="none"`` in the YAML.  The encoder applies log1p
    internally before the MLP; the raw counts stay as the NB reconstruction
    target and as the source of the observed library size
    (``ℓ = sum(x, dim=-1)``).

    Dispersion θ is learned per-gene (not per-cell).

    Fields:
      * ``latent_dim`` — Gaussian latent size.  Paper default 10; we default to 32
        to match the Exp-0 PCA sweep size.
      * ``encoder_hidden`` / ``decoder_hidden`` — MLP hidden widths (canonical
        scVI uses two 128-unit layers).
      * ``dropout_rate`` — applied after each hidden activation.
      * ``kl_weight``, ``kl_warmup_steps`` — β-VAE style linear warmup of the
        KL term; ``kl_weight=1`` after warmup is the canonical scVI ELBO.
    """

    latent_dim: int = 32
    encoder_hidden: list[int] = field(default_factory=lambda: [128, 128])
    decoder_hidden: list[int] = field(default_factory=lambda: [128, 128])
    dropout_rate: float = 0.1
    kl_weight: float = 1.0
    kl_warmup_steps: int = 2000

    # scANVI-style perturbation classifier head on the latent (Xu et al. 2021).
    # When True, adds q(pert_idx | z) parameterized as an MLP over z, and its
    # cross-entropy contribution is added to the total loss. The encoder does
    # NOT receive pert labels (z is read-out, not conditioned) — avoids the
    # "encoder cheats via label shortcut" failure mode.
    #
    # `num_perts` is resolved at runtime from the dataset and does not need to
    # be set in YAML. Must be positive when train_pert_classifier=True.
    train_pert_classifier: bool = False
    # Multiplies the classifier CE before adding to the ELBO. Default 1.0 —
    # likely needs tuning; reconstruction NLL is much larger in magnitude so
    # values in [1, 100] are reasonable starting points.
    classifier_weight: float = 1.0
    # Hidden widths of the classifier MLP. Empty list = single Linear layer
    # (latent_dim → num_perts). e.g. [128] adds one hidden layer with ReLU
    # + dropout_rate before the output Linear.
    classifier_hidden: list[int] = field(default_factory=list)
    num_perts: int | None = None

    # Perturbation-feature regression head (e.g. to ESM2 protein embeddings).
    # When True, adds a regressor head f(z) → R^pert_emb_dim trained with
    # cosine loss against batch["pert_emb"]. Unlike the classifier (which
    # injects discriminability), the regressor injects *relational* structure:
    # ESM2-similar perts are pulled toward similar z. Requires
    # dataset.pert_mapping_path (so batch["pert_emb"] flows in) and
    # pert_emb_dim > 0 (runtime-resolved).
    #
    # Cells whose pert has no entry in pert_mapping (e.g. SAFE_TARGET,
    # NO-TARGET) get zero pert_emb and are masked out of the regression loss.
    train_pert_regressor: bool = False
    regressor_weight: float = 1.0
    # Hidden widths of the regressor MLP. Empty list = single Linear layer
    # (latent_dim → pert_emb_dim).
    regressor_hidden: list[int] = field(default_factory=list)

    def validate(self) -> None:
        super().validate()
        if self.num_genes is None:
            raise ValueError(
                "ScVIConfig.num_genes must be resolved (set via "
                "set_attributes_from_dataset) before building the model."
            )
        if self.train_pert_classifier:
            if self.num_perts is None or self.num_perts <= 0:
                raise ValueError(
                    f"ScVIConfig.train_pert_classifier=True requires num_perts > 0; "
                    f"got num_perts={self.num_perts}. This is normally resolved "
                    f"from ds.num_perts at runtime."
                )
        if self.train_pert_regressor:
            if self.pert_emb_dim is None or self.pert_emb_dim <= 0:
                raise ValueError(
                    f"ScVIConfig.train_pert_regressor=True requires pert_emb_dim > 0; "
                    f"got pert_emb_dim={self.pert_emb_dim}. Set "
                    f"dataset.pert_mapping_path in your YAML (e.g. to "
                    f"datasets/competition_support_set/ESM2_pert_features.pt)."
                )

    def set_attributes_from_dataset(self, **kwargs) -> None:
        super().set_attributes_from_dataset(**kwargs)
        # Bind scVI-specific runtime-resolved field (ignored by base).
        if kwargs.get("num_perts") is not None:
            self.num_perts = int(kwargs["num_perts"])
