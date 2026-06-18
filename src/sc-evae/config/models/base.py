"""Base ModelConfig (ChoiceRegistry discriminated union) + draccus encode hook."""

from dataclasses import dataclass

import draccus
from draccus import ChoiceRegistry
from draccus.parsers.encoding import encode_choice
from draccus.utils import is_choice_type


@dataclass
class ModelConfig(ChoiceRegistry):
    num_genes: int | None = None
    is_count_discrete: bool | None = None
    max_count: int | None = None
    # Set by ``set_attributes_from_dataset`` from ``ds.pert_emb_dim``. Only
    # meaningful when the dataset has a precomputed ``pert_emb`` (e.g. ESM2)
    # via ``DatasetConfig.pert_mapping_path``.
    pert_emb_dim: int | None = None
    # Number of unique perturbation labels in the dataset. Required when any
    # subclass uses ``pert_cond_source='learned'`` or trains a per-label head.
    # Resolved at runtime from ``ds.num_perts``.
    num_perts: int | None = None
    num_cell_types: int | None = None
    # Source of the perturbation conditioning input for any subclass with
    # ``do_pert_cond=True``. Two modes:
    #   'projected' — project ``batch['pert_emb']`` (precomputed, e.g. ESM2)
    #                 via ``nn.Linear(pert_emb_dim, d_model)``. Requires the
    #                 dataset to have ``pert_mapping_path`` set.
    #   'learned'   — look up ``batch['pert_idx']`` in a learnable
    #                 ``nn.Embedding(num_perts, d_model)``. No external pert
    #                 features needed.
    pert_cond_source: str = "projected"

    @classmethod
    def default_choice_name(cls) -> str | None:
        return "vae"

    def validate(self) -> None:
        """Validate cross-field constraints after data dependent fields are resolved."""
        if self.pert_cond_source not in ("projected", "learned"):
            raise ValueError(
                f"pert_cond_source must be 'projected' or 'learned', "
                f"got {self.pert_cond_source!r}."
            )
        if getattr(self, "do_pert_cond", False):
            ctx = type(self).__name__
            if self.pert_cond_source == "learned":
                if self.num_perts is None or self.num_perts <= 0:
                    raise ValueError(
                        f"{ctx}.do_pert_cond=True with pert_cond_source='learned' "
                        f"requires num_perts > 0; got num_perts={self.num_perts}. "
                        f"This is normally resolved from ds.num_perts at runtime."
                    )
            else:  # 'projected'
                if self.pert_emb_dim is None or self.pert_emb_dim <= 0:
                    raise ValueError(
                        f"{ctx}.do_pert_cond=True with pert_cond_source='projected' "
                        f"requires pert_emb_dim > 0; got pert_emb_dim={self.pert_emb_dim}. "
                        f"Set dataset.pert_mapping_path in your YAML, or use "
                        f"pert_cond_source='learned' for a learnable embedding."
                    )
        return None

    def set_attributes_from_dataset(
        self,
        *,
        num_genes: int,
        pert_emb_dim: int | None = None,
        count_max: int | None = None,
        num_cell_types: int | None = None,
        num_perts: int | None = None,
    ) -> None:
        """
        Bind dataset-derived attributes onto the model config in-place.

        Subclasses can override this to apply model-specific rules (e.g. writing
        ``count_max`` into an output-head-specific ``max_count`` field).
        ``num_perts`` is bound on the base since multiple subclasses now need it
        (learnable pert-cond embeddings, per-label heads).
        """
        self.num_genes = num_genes
        if count_max is not None:
            self.max_count = count_max
        if pert_emb_dim is not None:
            self.pert_emb_dim = pert_emb_dim
        if num_cell_types is not None:
            self.num_cell_types = num_cell_types
        if num_perts is not None:
            self.num_perts = int(num_perts)


@draccus.encode.register(ModelConfig, include_subclasses=True)
def _encode_model_config(obj, declared_type=None):
    """Always emit the ChoiceRegistry ``type:`` discriminator when dumping.

    draccus's default encode path only writes the discriminator when the
    *declared* field type is itself a choice type (see
    ``draccus/parsers/encoding.py``: ``encode_choice`` vs ``encode_dataclass``).
    When a nested model config is typed as ``Any`` (to avoid argparse
    ``--help`` recursion through the ``ModelConfig`` ChoiceRegistry), this hook
    ensures the ``type:`` discriminator is still emitted so the run's
    ``config.yaml`` round-trips and is reloadable.
    """
    if declared_type is not None and is_choice_type(declared_type):
        return encode_choice(obj, declared_type)
    for base in type(obj).__mro__:
        if base is not type(obj) and is_choice_type(base):
            return encode_choice(obj, base)
    raise TypeError(f"No ChoiceRegistry ancestor found for {type(obj).__name__}")
