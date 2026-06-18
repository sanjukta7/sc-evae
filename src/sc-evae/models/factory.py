import json
import os

import torch
import torch.nn as nn

from sc_evae.config.models import (
    AutoregressiveModelConfig,
    sc_evaeConfig,
    FlowMatchingModelConfig,
    MaskedDiffusionModelConfig,
    ModelConfig,
    ScVIConfig,
)
from sc_evae.models.vae import sc_evae


def make_model(
    model_config: ModelConfig,
    log_lib_prior_mu: float = 0.0,
    log_lib_prior_std: float = 1.0,
) -> nn.Module:
    """
    Build a model instance from a model config.

    Parameters
    ----------
    log_lib_prior_mu, log_lib_prior_std:
        Parameters of the log-normal prior over library size (total UMI count).
        Set from ``DataStats.log_lib_mean`` / ``log_lib_std`` on the training split.
        Stored as non-trainable buffers in the model checkpoint and used at
        generation time (when no input cell is available) to sample l_n.
        Unused for AutoregressiveModelConfig.
    """
    if isinstance(model_config, sc_evaeConfig):
        return sc_evae(
            model_config,
            log_lib_prior_mu=log_lib_prior_mu,
            log_lib_prior_std=log_lib_prior_std,
        )

    if isinstance(model_config, ScVIConfig):
        from sc_evae.models.scvi import ScVI

        return ScVI(
            model_config,
            log_lib_prior_mu=log_lib_prior_mu,
            log_lib_prior_std=log_lib_prior_std,
        )

    if isinstance(model_config, AutoregressiveModelConfig):
        from sc_evae.models.autoregressive import AutoregressiveModel

        vae = None
        if model_config.vae_path is not None:
            vae = load_vae_from_run(model_config.vae_path)
        return AutoregressiveModel(model_config, vae=vae)

    if isinstance(model_config, MaskedDiffusionModelConfig):
        from sc_evae.models.masked_diffusion import MaskedDiffusionModel

        vae = None
        if model_config.vae_path is not None:
            vae = load_vae_from_run(model_config.vae_path)
        return MaskedDiffusionModel(model_config, vae=vae)

    if isinstance(model_config, FlowMatchingModelConfig):
        from sc_evae.models.flow_matching import FlowMatchingModel

        vae = None
        if model_config.vae_path is not None:
            vae = load_vae_from_run(model_config.vae_path)
        return FlowMatchingModel(model_config, vae=vae)

    raise NotImplementedError(
        f"Model config type {type(model_config).__name__} is not supported yet."
    )


# ---------------------------------------------------------------------------
# VAE loading utilities
# ---------------------------------------------------------------------------


def load_weights_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str | torch.device = "cpu",
) -> None:
    """
    Load model weights from an Accelerate checkpoint directory.

    Tries ``model.safetensors`` first (Accelerate's preferred format), then
    falls back to ``pytorch_model.bin``. Strips the ``module.`` prefix added
    by DDP wrappers.

    Parameters
    ----------
    model          : The model to load weights into (in-place).
    checkpoint_path: Path to the Accelerate checkpoint directory.
    device         : Device to map tensors to.
    """
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file

        state_dict = load_file(safetensors_path, device=str(device))
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location=device, weights_only=True)
    else:
        raise FileNotFoundError(
            f"No model weights found in {checkpoint_path!r}. "
            "Expected model.safetensors or pytorch_model.bin."
        )

    # Strip DDP 'module.' prefix if present
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Safetensors dedupes tied/aliased parameters at save time (when two
    # parameter names resolve to the same underlying Parameter, only one is
    # written). Mirror saved tensors into any alias that resolves to the same
    # Parameter in the live model so strict ``load_state_dict`` succeeds.
    storage_to_names: dict[int, list[str]] = {}
    for name, p in model.state_dict().items():
        storage_to_names.setdefault(p.data_ptr(), []).append(name)
    for names in storage_to_names.values():
        present = [n for n in names if n in state_dict]
        missing = [n for n in names if n not in state_dict]
        if present and missing:
            for m in missing:
                state_dict[m] = state_dict[present[0]]

    model.load_state_dict(state_dict)


def load_vae_from_run(
    vae_path: str,
    device: str | torch.device = "cpu",
) -> sc_evae:
    """
    Load a frozen sc_evae from a completed training run directory.

    Reads ``{vae_path}/config.yaml`` to reconstruct the model and
    ``{vae_path}/dataset_stats.json`` for the library-size prior. Weights
    are loaded from the latest checkpoint under ``{vae_path}/checkpoints/``.

    The returned VAE has all parameters frozen (``requires_grad=False``) and
    is placed in eval mode.

    Parameters
    ----------
    vae_path : Path to the experiment directory written by ``scripts/train.py``
               (contains config.yaml, dataset_stats.json, checkpoints/).
    device   : Device to load the model onto.

    Returns
    -------
    sc_evae with frozen parameters.
    """
    import draccus

    from sc_evae.config.training import TrainConfig
    from sc_evae.training.train_utils import latest_checkpoint

    # --- Load config --------------------------------------------------------
    config_path = os.path.join(vae_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"VAE config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        cfg = draccus.load(TrainConfig, f)

    if not isinstance(cfg.model, sc_evaeConfig):
        raise TypeError(
            f"Expected an sc_evaeConfig at {config_path}, "
            f"got {type(cfg.model).__name__}."
        )

    # --- Load library-size prior from dataset stats -------------------------
    stats_path = os.path.join(vae_path, "dataset_stats.json")
    log_lib_mu, log_lib_std = 0.0, 1.0
    if os.path.exists(stats_path):
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        log_lib_mu = stats.get("log_lib_mean", 0.0)
        log_lib_std = stats.get("log_lib_std", 1.0)

    # --- Build model --------------------------------------------------------
    vae = sc_evae(
        cfg.model, log_lib_prior_mu=log_lib_mu, log_lib_prior_std=log_lib_std
    )

    # --- Load weights -------------------------------------------------------
    checkpoint_dir = os.path.join(vae_path, "checkpoints")
    ckpt_path = latest_checkpoint(checkpoint_dir)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir!r}. "
            "Train the VAE first before using it as an AR prior."
        )
    load_weights_from_checkpoint(vae, ckpt_path, device=device)

    # --- Freeze and eval ----------------------------------------------------
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    vae = vae.to(device)

    return vae
