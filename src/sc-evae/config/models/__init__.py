"""Model configs (draccus ChoiceRegistry dataclasses).

Split out of the former monolithic ``config/model.py`` to mirror the per-model
layout under ``src/sc_evae/models/``. Submodules are imported here in
dependency / registration order so that importing ``sc_evae.config.models``
(or anything from it) registers every ``ModelConfig`` / ``QuantizerConfig`` /
``OutputHeadConfig`` / ``InputHeadConfig`` subclass with draccus.

Import the symbols you need from this package, e.g.::

    from sc_evae.config.models import sc_evaeConfig, FSQConfig
"""

# --- leaf configs (no intra-package deps) ---
from sc_evae.config.models.transformer import TransformerConfig
from sc_evae.config.models.quantizer import (
    FSQConfig,
    LFQConfig,
    NoQuantConfig,
    QuantizerConfig,
    VQConfig,
    get_fsq_level,
)
from sc_evae.config.models.heads import (
    CrossEntropyHeadConfig,
    EmbeddingInputHeadConfig,
    HurdleHeadConfig,
    InputHeadConfig,
    MLPInputHeadConfig,
    MSEHeadConfig,
    NBHeadConfig,
    OutputHeadConfig,
    SinusoidalInputHeadConfig,
    _CONTINUOUS_HEAD_TYPES,
    _DISCRETE_HEAD_TYPES,
)

# --- base ModelConfig + draccus encode hook (must precede subclasses) ---
from sc_evae.config.models.base import ModelConfig

# --- concrete model configs (register ModelConfig subclasses on import) ---
from sc_evae.config.models.scvi import ScVIConfig
from sc_evae.config.models.vae import sc_evaeConfig
from sc_evae.config.models.autoregressive import AutoregressiveModelConfig
from sc_evae.config.models.masked_diffusion import MaskedDiffusionModelConfig
from sc_evae.config.models.flow_matching import FlowMatchingModelConfig

__all__ = [
    "TransformerConfig",
    "QuantizerConfig",
    "VQConfig",
    "FSQConfig",
    "LFQConfig",
    "NoQuantConfig",
    "get_fsq_level",
    "OutputHeadConfig",
    "MSEHeadConfig",
    "HurdleHeadConfig",
    "NBHeadConfig",
    "CrossEntropyHeadConfig",
    "InputHeadConfig",
    "MLPInputHeadConfig",
    "SinusoidalInputHeadConfig",
    "EmbeddingInputHeadConfig",
    "_DISCRETE_HEAD_TYPES",
    "_CONTINUOUS_HEAD_TYPES",
    "ModelConfig",
    "ScVIConfig",
    "sc_evaeConfig",
    "AutoregressiveModelConfig",
    "MaskedDiffusionModelConfig",
    "FlowMatchingModelConfig",
]
