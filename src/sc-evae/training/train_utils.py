from __future__ import annotations

import logging
import math
import os
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from sc_evae.config.dataset import DatasetConfig
from sc_evae.config.training import TrainConfig
from sc_evae.training.tokenizer import _apply_binning, bin_id_to_value

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class ColorFormatter(logging.Formatter):
    """Formatter that wraps the message in ANSI color codes by log level.

    Intended for terminal (stream) handlers only — file handlers should use a
    plain ``logging.Formatter`` so log files stay free of escape sequences.
    """

    _RESET = "\x1b[0m"
    _COLORS = {
        logging.DEBUG: "\x1b[36m",  # cyan
        logging.INFO: "",  # default terminal color
        logging.WARNING: "\x1b[33m",  # yellow
        logging.ERROR: "\x1b[31m",  # red
        logging.CRITICAL: "\x1b[1;31m",  # bold red
    }

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        color = self._COLORS.get(record.levelno, "")
        return f"{color}{msg}{self._RESET}" if color else msg


def setup_logging(
    log_dir: str,
    *,
    is_main_process: bool,
    log_filename: str = "train.log",
    level: int = logging.INFO,
    fmt: str = "%(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Configure root logging with colored console output and a plain log file.

    - Console: ``ColorFormatter`` (yellow warnings, red errors, etc.).
    - File:    plain ``logging.Formatter``, only attached on the main process.
    - Silences noisy ``accelerate.checkpointing`` / ``accelerate.accelerator``
      loggers at WARNING level.
    """
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ColorFormatter(fmt, datefmt=datefmt))

    if is_main_process:
        file_handler: logging.Handler = logging.FileHandler(
            os.path.join(log_dir, log_filename)
        )
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    else:
        file_handler = logging.NullHandler()

    logging.basicConfig(
        level=level,
        handlers=[stream_handler, file_handler],
        force=True,
    )
    logging.getLogger("accelerate.checkpointing").setLevel(logging.WARNING)
    logging.getLogger("accelerate.accelerator").setLevel(logging.WARNING)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> LambdaLR:
    warmup = cfg.warmup_steps
    total = cfg.n_iters

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        if cfg.lr_schedule == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        elif cfg.lr_schedule == "linear":
            return max(0.0, 1.0 - progress)
        else:  # "constant"
            return 1.0

    return LambdaLR(optimizer, lr_lambda)


def latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Return path to the most recently saved accelerate checkpoint, or None."""
    latest_file = os.path.join(checkpoint_dir, "latest")
    if os.path.exists(latest_file):
        with open(latest_file) as f:
            name = f.read().strip()
        path = os.path.join(checkpoint_dir, name)
        if os.path.isdir(path):
            return path
    return None


def save_latest_pointer(checkpoint_dir: str, name: str) -> None:
    with open(os.path.join(checkpoint_dir, "latest"), "w") as f:
        f.write(name)


def save_checkpoint(
    accelerator: Accelerator, checkpoint_dir: str, checkpoint_name: str
) -> str:
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    accelerator.save_state(checkpoint_path)
    if accelerator.is_main_process:
        save_latest_pointer(checkpoint_dir, checkpoint_name)
    return checkpoint_path


def log_checkpoint_artifact(
    ckpt_path: str,
    run_name: str,
    step: int,
    *,
    extra_aliases: list[str] | None = None,
) -> None:
    """Upload an accelerate checkpoint directory as a W&B artifact.

    Versioned under ``f"{run_name}-ckpt"`` with aliases ``["latest", "step_{step}"]``
    plus any in ``extra_aliases``. Non-blocking — wandb uploads in the background.
    """
    import wandb

    art = wandb.Artifact(name=f"{run_name}-ckpt", type="model", metadata={"step": step})
    art.add_dir(ckpt_path)
    aliases = ["latest", f"step_{step:07d}"] + (extra_aliases or [])
    wandb.log_artifact(art, aliases=aliases)


def to_scalar_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        out = value.detach()
        if out.numel() == 0:
            raise ValueError("Metric tensor must have at least one element.")
        if out.numel() != 1:
            out = out.float().mean()
        return out.float().reshape(())

    if isinstance(value, (int, float, bool)):
        return torch.tensor(float(value), device=device)

    raise TypeError(f"Unsupported metric type: {type(value).__name__}")


def reduce_scalar_mean(accelerator: Accelerator, value: Any) -> float:
    scalar = to_scalar_tensor(value, accelerator.device)
    return accelerator.gather(scalar).float().mean().item()


def reduce_metric_dict(
    accelerator: Accelerator, metrics: dict[str, Any]
) -> dict[str, float]:
    reduced: dict[str, float] = {}
    for name, value in metrics.items():
        reduced[name] = reduce_scalar_mean(accelerator, value)
    return reduced


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{k}": v for k, v in metrics.items()}


def resolve_run_name(
    run_name: str,
    work_dir: str,
    prefix_run_name_with_datetime: bool,
    accelerator: Accelerator,
) -> str:
    should_prefix = False
    if accelerator.is_main_process:
        run_name_exists = os.path.isdir(os.path.join(work_dir, run_name))
        should_prefix = prefix_run_name_with_datetime and run_name_exists

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        holder = [should_prefix if accelerator.is_main_process else False]
        torch.distributed.broadcast_object_list(holder, src=0)
        should_prefix = bool(holder[0])

    resolved_run_name = run_name
    if should_prefix and accelerator.is_main_process:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        resolved_run_name = f"{ts}_{run_name}"

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        holder = [resolved_run_name if accelerator.is_main_process else ""]
        torch.distributed.broadcast_object_list(holder, src=0)
        return holder[0]

    return resolved_run_name


def load_split_spec(split_path: str) -> tuple[list[str], list[str]]:
    """
    Load and validate a cell-type × perturbation holdout spec from JSON.

    The file must be a JSON object of shape::

        {"cell_type": ["hepg2"], "pert": ["ABCB7", "ABHD11", ...]}

    Both keys are required and must hold non-empty lists of strings. The
    held-out test set is the cross product ``cell_type`` × ``pert`` (see
    ``split_dataset_by_celltype_pert``).

    Returns
    -------
    (cell_types, perts) — the two lists, ready to pass straight to
    ``split_dataset_by_celltype_pert``.
    """
    import json

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split spec not found: {split_path!r}")
    with open(split_path, encoding="utf-8") as f:
        spec = json.load(f)

    if not isinstance(spec, dict):
        raise ValueError(
            f"Split spec {split_path!r} must be a JSON object, got {type(spec).__name__}."
        )
    for key in ("cell_type", "pert"):
        if key not in spec:
            raise ValueError(
                f"Split spec {split_path!r} is missing required key {key!r}. "
                f'Expected {{"cell_type": [...], "pert": [...]}}.'
            )
        if not isinstance(spec[key], list) or not spec[key]:
            raise ValueError(
                f"Split spec {split_path!r} key {key!r} must be a non-empty list, "
                f"got {spec[key]!r}."
            )
    return list(spec["cell_type"]), list(spec["pert"])


def resolve_train_val_split(
    ds: Any,
    dataset_cfg: DatasetConfig,
    seed: int,
) -> tuple[Any, Any]:
    """
    Single dispatch point for building the train/val split.

    Selects between the random / pert-frac split (default) and a cell-type ×
    perturbation holdout described by a JSON spec at
    ``DatasetConfig.split_path``. Both ``scripts/train.py`` and
    ``scripts/eval.py`` route through this helper so they cannot drift apart.

    Modes:
      * ``split_path is None`` — fallback to
        ``split_dataset(val_frac, val_pert_frac, …)``.
      * ``split_path`` set — load the JSON holdout spec (see
        ``load_split_spec``) and hold out its ``cell_type`` × ``pert`` cross
        product via ``split_dataset_by_celltype_pert``. The released scLDM
        splits live in ``assets/`` (e.g. ``assets/scldm_replogle_test.json``).

    Returns
    -------
    (train_ds, val_ds) — both are ``PerturbationDataset`` views over the
    same underlying arrays.
    """
    from sc_evae.training.data_loader import (
        split_dataset,
        split_dataset_by_celltype_pert,
    )

    logger = logging.getLogger(__name__)
    split_path = dataset_cfg.split_path

    if split_path is None:
        logger.info(
            "Train/val split: random (val_frac=%.3f, val_pert_frac=%.3f, seed=%d)",
            dataset_cfg.val_frac,
            dataset_cfg.val_pert_frac,
            seed,
        )
        return split_dataset(
            ds,
            val_frac=dataset_cfg.val_frac,
            val_pert_frac=dataset_cfg.val_pert_frac,
            seed=seed,
            ctrl_label=dataset_cfg.ctrl_label,
            ctrl_labels=dataset_cfg.ctrl_labels or None,
        )

    test_cell_types, test_perts = load_split_spec(split_path)
    logger.info(
        "Train/val split: %s (cell_type=%r × %d perts held out)",
        split_path,
        test_cell_types,
        len(test_perts),
    )
    return split_dataset_by_celltype_pert(
        ds,
        test_cell_types=test_cell_types,
        test_perts=test_perts,
    )


def resolve_config(cfg: TrainConfig, ds: Any) -> TrainConfig:
    """
    Single entry point for fully resolving TrainConfig before model build.
    This function binds dataset-derived attributes on ``cfg.model`` via
       ``ModelConfig.set_attributes_from_dataset(...)``.
    """
    cfg.model.set_attributes_from_dataset(
        num_genes=ds.num_genes,
        pert_emb_dim=getattr(ds, "pert_emb_dim", None),
        count_max=getattr(ds, "count_max", None),
        num_cell_types=getattr(ds, "num_cell_types", None),
        num_perts=getattr(ds, "num_perts", None),
    )
    cfg.model.validate()
    return cfg


def to_metric_space(
    x: torch.Tensor,
    dataset_cfg: DatasetConfig,
    library_size: torch.Tensor | None = None,
    hvg_library: torch.Tensor | None = None,
    basis: str = "full_library",
) -> torch.Tensor:
    """
    Convert model outputs / ground-truth counts to a common log1p metric space.

    Different output heads (log1p, raw counts, quantile bins) produce values in
    different spaces; this maps them all to one log1p space so the perturbation
    metrics are comparable and follow the GEARS / scLDM / cell-eval convention.
    The exact space depends on ``basis``:
      * ``hvg_row_sum`` (eval default) — log1p of per-cell **HVG-fractional
        CP10K** (each cell rescaled to sum 1e4 across the HVG panel). This is
        config-, library-, and target-invariant — every output head collapses
        to the same result — and matches scLDM / cell-eval.
      * ``full_library`` — log1p of **counts-per-target** on the full panel,
        where "target" is ``dataset.target_sum`` (or the per-file median when
        ``None``). This is NOT literally CP10K for median-normalized data, and
        its absolute scale is config-dependent, so it is only safe for
        scale-invariant metrics (the train-time proxy uses it).

    When ``dataset_cfg.use_binning != "none"``, ``x`` is expected to be bin
    indices (predicted argmax IDs or training-data bin labels).  Bin IDs are
    first mapped back to representative continuous values via
    ``bin_id_to_value`` and then routed through the appropriate preprocessing
    chain.

    Parameters
    ----------
    x : (..., G) tensor — predictions or ground-truth counts in training data space.
    dataset_cfg : the DatasetConfig used to load the data.
    library_size : optional ``(B,)`` or ``(B, 1)`` tensor of full pre-HVG
        library sizes (provided by the data loader as ``batch["library_size"]``).
        Used as the CP10K denominator under ``basis="full_library"`` for
        raw-counts data. Ignored for log1p- or apply_normalize-baked data
        (the dataset has already baked the full library into the values).
    hvg_library : currently ignored (kept for API stability — the
        ``hvg_row_sum`` conversion uses each cell's own row sum directly).
    basis : ``"full_library"`` or ``"hvg_row_sum"``. Selects what the output
        CP10K is normalized against.
        - ``"full_library"`` (default) — absolute CP10K against the full
          pre-HVG library. Matches scVI / scanpy convention.
        - ``"hvg_row_sum"`` — HVG-fractional CP10K, computed as each cell's
          per-gene CP10K rescaled to sum to 1e4 across HVGs. Matches
          scLDM / State paper convCaention (their pipeline ships pre-HVG-
          sliced files, so their library_size IS the HVG row sum and their
          CP10K naturally sums to 1e4 per cell across HVGs).

    The ``hvg_row_sum`` conversion is computed from each cell's own row sum
    (``cpx_full.sum(-1)``), so it works uniformly for log1p-trained,
    apply_normalize-trained, and raw-counts-trained models without needing
    an external library size argument.

    Returns
    -------
    (..., G) tensor in the log1p metric space selected by ``basis`` (see above).
    """
    if basis not in ("full_library", "hvg_row_sum"):
        raise ValueError(
            f"basis must be 'full_library' or 'hvg_row_sum', got {basis!r}"
        )

    if dataset_cfg.use_binning == "quantile":
        # Quantile bins are integer counts — fall through to the chain below.
        x = bin_id_to_value(x, "quantile")

    # Step 1: build ``cpx_full`` — counts-per-target on the FULL-LIBRARY basis.
    # "cpx" = counts per X, where X is the per-cell normalization target: 1e4
    # for the raw-count branch (literal CP10K), or the loader's target_sum /
    # per-file median for the log1p / apply_normalize branches. The hvg_row_sum
    # basis below divides X out, so it only matters for the full_library return.
    fallback_to_xsum = False
    if dataset_cfg.apply_log1p:
        # Stored: log1p(raw · target / full_lib_dl). expm1 recovers the
        # normalized counts in the loader's "counts-per-target" space (target =
        # config.target_sum, or the per-file median library size when None).
        cpx_full = torch.expm1(x)
    elif dataset_cfg.apply_normalize:
        # Stored: raw · target / full_lib_dl (same counts-per-target space).
        cpx_full = x
    elif library_size is not None:
        lib = library_size.to(x.device, dtype=x.dtype).clamp_min(1.0)
        if lib.ndim == 1:
            lib = lib.unsqueeze(-1)
        cpx_full = x / lib * 1e4
    else:
        # Raw counts, no library_size — fall back to x.sum(-1), which is the
        # HVG row sum when HVG slicing is active. Note: this means cpx_full
        # is actually HVG-fractional, not full-library; flagged via the
        # ``fallback_to_xsum`` flag so the basis conversion below knows.
        fallback_to_xsum = True
        if basis == "full_library" and (
            getattr(dataset_cfg, "num_hvg", None) is not None
            or getattr(dataset_cfg, "gene_order_path", None) is not None
        ):
            import warnings as _warnings

            _warnings.warn(
                "to_metric_space: basis='full_library' but library_size not "
                "passed with HVG slicing active — falling back to x.sum(-1) "
                "(HVG row sum). Result is HVG-fractional, not absolute. Pass "
                'library_size=batch["library_size"] for true absolute CP10K, '
                "or set basis='hvg_row_sum' explicitly.",
                RuntimeWarning,
                stacklevel=2,
            )
        cpx_full = x / x.sum(dim=-1, keepdim=True).clamp_min(1.0) * 1e4

    # Step 2: convert to requested basis.
    if basis == "full_library" or fallback_to_xsum:
        # full_library basis: cpx_full is what we want.
        # fallback_to_xsum: cpx_full is already HVG-fractional (we
        # divided by x.sum(-1)), which is the same as the hvg_row_sum
        # result. Either way, log1p it.
        return torch.log1p(cpx_full)

    # basis == "hvg_row_sum": rescale each cell so its count vector sums to
    # 1e4 across the HVG dimension (→ HVG-fractional CP10K). The per-cell
    # rescale divides out cpx_full's target X, so every output-head branch
    # collapses to log1p(raw_g / HVG_row_sum * 1e4) — config-, library-, and
    # target-invariant. Computed from the cell's own row sum, so it needs no
    # external library_size / hvg_library and never mismatches when
    # batch["counts"] is in a different space (e.g. log1p-baked). Matches
    # scLDM's / cell-eval's per-cell convention exactly.
    # Clamp at 0 first: MSE-trained models can predict slightly-negative
    # log1p values near zero, which would make expm1 negative and break
    # the row-sum normalisation.
    cpx_clamped = cpx_full.clamp_min(0.0)
    row_sum = cpx_clamped.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.log1p(cpx_clamped * (1e4 / row_sum))


def run_validation(
    model: nn.Module,
    val_dl: Any,
    accelerator: Accelerator,
    deg_table: dict,
    unique_perts: list[str],
    dataset_cfg: DatasetConfig,
    max_batches: int | None = None,
) -> dict[str, float]:
    """
    Run one full validation pass and return a flat metrics dict.

    Computes:
      - All model-reported losses/metrics (averaged over batches)
      - proxy_pearson_delta_deg20 and proxy_r2_deg20 in log1p(CP10K) space

    Parameters
    ----------
    model        : accelerator-wrapped model (called for loss/metrics).
    val_dl       : validation DataLoader (accelerator-prepared).
    accelerator  : Accelerator instance.
    deg_table    : precomputed DEG table (from build_deg_table).
    unique_perts : ordered list of perturbation labels (ds.unique_perts).
    dataset_cfg  : DatasetConfig used to load the data — controls to_metric_space.

    Returns
    -------
    dict[str, float] with keys like "loss", "recon_loss", and the train-time
    proxy metrics "proxy_pearson_delta_deg20", "proxy_r2_deg20", …

    NOTE: the ``proxy_*`` keys are fast, per-batch single-cell approximations
    computed during training for monitoring only. They are NOT the paper's
    reported numbers — those come from the pseudobulk pipeline in
    ``scripts/eval.py`` (distribution + cell-eval metrics). Keep the two
    mentally separate; the proxy is a noisy training signal, not an eval.
    """
    from sc_evae.metrics.perturbation import (
        batch_pearson_delta_deg20,
        batch_r2_deg20,
    )

    unwrapped = accelerator.unwrap_model(model)
    metric_sums: dict[str, float] = {}
    n_batches = 0
    # Train-time PROXY perturbation signal (noisy, single-cell, monitoring only).
    # Two DEG-focused axes: delta correlation (direction/shape of the effect)
    # and R² (magnitude). The raw top-20 Pearson (baseline-inflated by
    # housekeeping) and the all-genes delta (dominated by the unchanged
    # majority) were dropped — the rigorous, properly-pseudobulked deg20 /
    # delta / R² numbers come from scripts/eval.py.
    proxy_names = (
        "proxy_pearson_delta_deg20",  # delta correlation on the top-20 DEGs
        "proxy_r2_deg20",  # magnitude (R²) on the top-20 DEGs
    )
    proxy_sums: dict[str, float] = {n: 0.0 for n in proxy_names}
    proxy_counts: dict[str, int] = {n: 0 for n in proxy_names}

    val_iter = (
        tqdm(val_dl, desc="val", leave=False) if accelerator.is_main_process else val_dl
    )
    with torch.no_grad():
        for batch in val_iter:
            if max_batches is not None and n_batches >= max_batches:
                break
            with accelerator.autocast():
                val_loss, val_metrics = model(batch)
            reduced = reduce_metric_dict(accelerator, {"loss": val_loss, **val_metrics})
            for k, v in reduced.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v
            n_batches += 1

            with accelerator.autocast():
                pred = unwrapped.pseudo_sample(batch)
            pert_keys = [unique_perts[int(i)] for i in batch["pert_idx"].cpu()]
            lib = batch.get("library_size")
            lib_cpu = lib.cpu() if lib is not None else None
            pred_m = to_metric_space(
                pred.detach().cpu(), dataset_cfg, library_size=lib_cpu
            )
            counts_m = to_metric_space(
                batch["counts"].cpu(), dataset_cfg, library_size=lib_cpu
            )

            # Train-time PROXY metrics only (not the paper's eval numbers) —
            # delta correlation + R² on the per-pert top-20 DEGs.
            updates = {
                "proxy_pearson_delta_deg20": batch_pearson_delta_deg20(
                    pred_m, counts_m, pert_keys, deg_table
                ),
                "proxy_r2_deg20": batch_r2_deg20(
                    pred_m, counts_m, pert_keys, deg_table
                ),
            }
            for name, val in updates.items():
                if not math.isnan(val):
                    proxy_sums[name] += val
                    proxy_counts[name] += 1

    results = {k: v / max(1, n_batches) for k, v in metric_sums.items()}
    for name in proxy_names:
        if proxy_counts[name] > 0:
            results[name] = proxy_sums[name] / proxy_counts[name]
    return results


def build_deg_table_from_dataset(
    ds: Any,
    dataset_cfg: DatasetConfig,
    top_k: int = 20,
) -> dict:
    """
    Build a DEG table from all cells in a PerturbationDataset.

    Uses all cells (train + val) so held-out perturbations also get a DEG
    mask.  The table is only used for gene selection, not as prediction
    targets, so including val cells does not leak label information.

    In lazy mode the full matrix is never materialised — per-perturbation
    sums are accumulated in chunks directly from HDF5.

    Parameters
    ----------
    ds          : PerturbationDataset (full, unsplit dataset).
    dataset_cfg : DatasetConfig — controls to_metric_space conversion.
    top_k       : number of top DEGs to keep per perturbation (default 20).

    Returns
    -------
    dict mapping pert_label → {"mask", "indices", "ctrl_mean", "pert_mean"}
    """
    from sc_evae.metrics.deg_table import build_deg_table

    all_pert_labels = [ds.unique_perts[int(i)] for i in ds._pert_indices]

    ctrl_set: set[str] = (
        set(dataset_cfg.ctrl_labels)
        if dataset_cfg.ctrl_labels
        else {dataset_cfg.ctrl_label}
    )
    if all(lbl in ctrl_set for lbl in all_pert_labels):
        import logging

        logging.getLogger(__name__).info(
            "All %d cells are ctrl (labels=%s) — skipping DEG build; returning empty table.",
            len(all_pert_labels),
            sorted(ctrl_set),
        )
        return {}

    if not ds._load_lazily:
        # Eager: full-panel library size was captured pre-normalize_total in
        # _init_eager. Pass it so to_metric_space's raw-counts branch (used
        # when apply_normalize=False AND apply_log1p=False) divides by the
        # full library, matching scVI/scLDM/State CP10K convention.
        lib_t = (
            torch.from_numpy(ds._library_size) if ds._library_size is not None else None
        )
        all_counts_metric = to_metric_space(
            torch.from_numpy(ds.counts),
            dataset_cfg,
            library_size=lib_t,
        ).numpy()
        return build_deg_table(
            all_counts_metric,
            all_pert_labels,
            ctrl_label=dataset_cfg.ctrl_label,
            top_k=top_k,
        )

    # --- lazy path: stream HDF5 in chunks, accumulate per-pert sums ----------
    import anndata
    import numpy as np
    import scipy.sparse

    n_cells = len(ds._pert_indices)
    n_genes = ds.num_genes
    top_k = min(top_k, n_genes)
    chunk_size = 4096

    # Determine ctrl labels (single or multi)
    ctrl_set: set[str] = set()
    if dataset_cfg.ctrl_labels:
        ctrl_set = set(dataset_cfg.ctrl_labels)
    else:
        ctrl_set = {dataset_cfg.ctrl_label}

    # Per-pert accumulators: sum (G,) and count (scalar)
    pert_sum: dict[str, np.ndarray] = {}
    pert_count: dict[str, int] = {}

    # Iterate per source file so physical row indices stay within bounds.
    h5ad_paths = ds._h5ad_paths
    obs_per_file = ds._obs_int_indices_per_file
    offsets = ds._file_row_offsets  # length n_files + 1, global-row boundaries

    n_chunks_total = 0
    for f_idx in range(len(h5ad_paths)):
        n_file_cells = int(offsets[f_idx + 1] - offsets[f_idx])
        n_chunks_total += (n_file_cells + chunk_size - 1) // chunk_size

    pbar = tqdm(total=n_chunks_total, desc="building DEG table")
    for f_idx, path in enumerate(h5ad_paths):
        file_start = int(offsets[f_idx])
        file_end = int(offsets[f_idx + 1])
        n_file_cells = file_end - file_start
        phys_file = obs_per_file[f_idx]

        tmp_adata = anndata.read_h5ad(path, backed="r")
        X = tmp_adata.X

        # When the file's top-N HVGs are all present in its native panel,
        # we can fancy-index sparse columns directly to the HVG order and
        # densify only the (chunk_size, num_hvg) slice — skipping the
        # ~n_native-wide intermediate. Common case for parse single-file and
        # the Replogle combined-intersection ordering.
        hvg_no_missing = (
            ds._hvg_no_missing_per_file is not None
            and ds._hvg_adjusted_per_file is not None
            and ds._hvg_no_missing_per_file[f_idx]
        )
        hvg_adj = ds._hvg_adjusted_per_file[f_idx] if hvg_no_missing else None

        for chunk_start in range(0, n_file_cells, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_file_cells)
            chunk_phys = phys_file[chunk_start:chunk_end]

            sort_order = np.argsort(chunk_phys)
            rows_raw = X[chunk_phys[sort_order]]

            # Full-panel library size, computed BEFORE any HVG slicing so
            # ``apply_normalize`` divides by the true library size — same as
            # ``sc.pp.normalize_total`` in the eager loader. Doing this after
            # HVG slicing (the previous behavior) divided by HVG-row-sum,
            # inflating CP10K by 1/(HVG share of library) — typically 1.5–5×.
            if scipy.sparse.issparse(rows_raw):
                full_row_sums = (
                    np.asarray(rows_raw.sum(axis=1)).ravel().astype(np.float64)
                )
            else:
                full_row_sums = (
                    np.asarray(rows_raw, dtype=np.float32)
                    .sum(axis=1)
                    .astype(np.float64)
                )

            if hvg_no_missing and scipy.sparse.issparse(rows_raw):
                # Fast path: sparse column fancy-index to HVG order, densify
                # only the small (chunk, num_hvg) slice. Memory: chunk × 2k
                # × 4B (~32 MB for Parse) instead of chunk × 40k × 4B (~660 MB).
                rows = np.asarray(
                    rows_raw[:, hvg_adj].toarray(),
                    dtype=np.float32,
                )
            else:
                # Slow path (only when target genes are missing from this
                # file's native panel): full densify, permute, slice.
                rows = rows_raw
                if scipy.sparse.issparse(rows):
                    rows = rows.toarray()
                rows = np.asarray(rows, dtype=np.float32)
                if ds._gene_perms_per_file is not None:
                    rows = ds._apply_permutation_dense(
                        rows,
                        ds._gene_perms_per_file[f_idx],
                        ds._n_native_per_file[f_idx],
                    )
                if ds._num_hvg is not None:
                    rows = rows[:, : ds._num_hvg]

            restore_order = np.argsort(sort_order)
            rows = rows[restore_order]
            full_row_sums = full_row_sums[restore_order]

            # Match the data_loader → to_metric_space pipeline so ctrl_mean
            # lives in the *same* per-config metric space as the eval-time
            # pred / true tensors. Normalize uses the FULL-panel library
            # size (computed pre-slice above) — equivalent to
            # ``sc.pp.normalize_total(adata, target_sum)`` followed by HVG
            # slicing. For binned configs this routes ctrl through bin
            # midpoints (and CP10K + log1p for quantile), not through smooth
            # log1p(CP10K).
            if dataset_cfg.apply_normalize:
                rs = np.where(full_row_sums < 1.0, 1.0, full_row_sums)[:, None]
                # Use the dataset's representative target (config.target_sum, or
                # the global median library size when None) so ctrl_mean lands
                # in the same metric space as the loader's normalized values.
                rows = (rows / rs * ds.normalization_target_sum).astype(np.float32)
            if dataset_cfg.apply_log1p:
                rows = np.log1p(rows).astype(np.float32)
            if dataset_cfg.use_binning != "none":
                rows = _apply_binning(rows, dataset_cfg.use_binning)
            rows = to_metric_space(
                torch.from_numpy(rows),
                dataset_cfg,
                library_size=torch.from_numpy(full_row_sums),
            ).numpy()

            chunk_labels = np.asarray(
                all_pert_labels[file_start + chunk_start : file_start + chunk_end]
            )
            # Vectorized per-pert accumulation: groupby-style. For Parse this
            # collapses ~4096 Python-level vector adds per chunk into ~91
            # (one per unique cytokine in the chunk).
            unique_in_chunk, inv = np.unique(chunk_labels, return_inverse=True)
            for u_idx, label in enumerate(unique_in_chunk):
                mask = inv == u_idx
                n_in_group = int(mask.sum())
                if n_in_group == 0:
                    continue
                if label not in pert_sum:
                    pert_sum[label] = np.zeros(n_genes, dtype=np.float64)
                    pert_count[label] = 0
                pert_sum[label] += rows[mask].sum(axis=0)
                pert_count[label] += n_in_group

            pbar.update(1)

        tmp_adata.file.close()
    pbar.close()

    # Compute ctrl mean (pool all ctrl labels)
    ctrl_sum = np.zeros(n_genes, dtype=np.float64)
    ctrl_count = 0
    for lbl in ctrl_set:
        if lbl in pert_sum:
            ctrl_sum += pert_sum[lbl]
            ctrl_count += pert_count[lbl]
    if ctrl_count == 0:
        raise ValueError(
            f"No cells found for ctrl_labels={ctrl_set}. "
            "Check that ctrl_labels matches your data."
        )
    ctrl_mean = (ctrl_sum / ctrl_count).astype(np.float32)
    ctrl_mean_t = torch.from_numpy(ctrl_mean)

    # Build table for non-ctrl perts
    table: dict[str, dict] = {}
    for label, s in pert_sum.items():
        if label in ctrl_set:
            continue
        cnt = pert_count[label]
        if cnt < 1:
            continue
        pert_mean = (s / cnt).astype(np.float32)
        delta = np.abs(pert_mean - ctrl_mean)
        sorted_idx = np.argsort(delta)
        top_indices = sorted_idx[-top_k:][::-1].copy()
        mask = torch.zeros(n_genes, dtype=torch.bool)
        mask[top_indices] = True
        table[label] = {
            "mask": mask,
            "indices": torch.from_numpy(top_indices),
            "ctrl_mean": ctrl_mean_t,
            "pert_mean": torch.from_numpy(pert_mean),
        }

    import logging

    logging.getLogger(__name__).info(
        "Built DEG table (lazy): %d perturbations, top_k=%d, n_genes=%d",
        len(table),
        top_k,
        n_genes,
    )
    return table


# Top-k DEGs kept per perturbation for the train-time proxy metrics.
DEG_TOP_K = 20


def resolve_deg_table(
    ds: Any,
    dataset_cfg: DatasetConfig,
    experiment_dir: str,
    *,
    logger: Any,
    is_main_process: bool,
    top_k: int = DEG_TOP_K,
) -> dict:
    """
    Resolve the per-perturbation DEG table used for train-time proxy metrics.

    Resolution order:
      1. ``DatasetConfig.deg_table_path`` (YAML override)
      2. ``constants.resolve_default_deg_table_path(...)`` (registered defaults)
      3. compute on the fly via ``build_deg_table_from_dataset`` (last resort)

    On disk-load we shape-check (mask width == ``ds.num_genes``, indices len ==
    ``top_k``); a mismatch raises so we fall back to recompute rather than
    silently using the wrong table (we DON'T check ctrl_label, preprocessing, or
    numeric values — ``to_metric_space`` normalizes those away). A freshly
    computed table is saved to ``{experiment_dir}/deg_table.pt`` (main process
    only).

    Parameters
    ----------
    ds              : the (full, unsplit) PerturbationDataset.
    dataset_cfg     : DatasetConfig (deg_table_path, h5ad path(s), ctrl_label).
    experiment_dir  : run directory to save a freshly computed table into.
    logger          : accelerate logger (accepts ``main_process_only=``).
    is_main_process : only the main process writes the table to disk.
    top_k           : number of top DEGs per perturbation.

    Returns
    -------
    dict mapping pert_label → {"mask", "indices", "ctrl_mean", "pert_mean"}.
    """
    from sc_evae.training.constants import resolve_default_deg_table_path

    cached_path = dataset_cfg.deg_table_path or resolve_default_deg_table_path(
        dataset_cfg.h5ad_path or None,
        list(dataset_cfg.h5ad_paths) if dataset_cfg.h5ad_paths else None,
    )
    deg_table: dict | None = None
    if cached_path and os.path.exists(cached_path):
        logger.info(f"Loading DEG table from {cached_path} …", main_process_only=True)
        try:
            cached = torch.load(cached_path, weights_only=False)
            # Shape check: any pert entry must have the right gene-axis size.
            sample_entry = next(iter(cached.values()))
            mask_w = sample_entry["mask"].shape[0]
            idx_len = sample_entry["indices"].shape[0]
            if mask_w != ds.num_genes:
                raise ValueError(
                    f"DEG-table mask width {mask_w} != ds.num_genes {ds.num_genes}; "
                    f"the cached table was built with a different gene panel "
                    f"(num_hvg / gene_order_path)."
                )
            if idx_len != top_k:
                raise ValueError(
                    f"DEG-table indices length {idx_len} != top_k {top_k}; "
                    f"the cached table was built with a different top_k."
                )
            deg_table = cached
            logger.info(
                f"  Loaded {len(deg_table)} perts (mask_w={mask_w}, top_k={idx_len}). "
                f"Skipping build_deg_table_from_dataset.",
                main_process_only=True,
            )
        except Exception as e:
            logger.warning(
                f"Failed to load cached DEG table from {cached_path}: {e}. "
                f"Falling back to recompute.",
                main_process_only=True,
            )
            deg_table = None

    if deg_table is None:
        logger.info(
            f"Building DEG table (ctrl_label={dataset_cfg.ctrl_label!r}, "
            f"top_k={top_k}) …",
            main_process_only=True,
        )
        deg_table = build_deg_table_from_dataset(ds, dataset_cfg, top_k=top_k)
        if is_main_process:
            run_deg_path = os.path.join(experiment_dir, "deg_table.pt")
            torch.save(deg_table, run_deg_path)
            logger.info(
                f"Saved DEG table ({len(deg_table)} perts) to {run_deg_path}",
                main_process_only=True,
            )
    return deg_table


def cycle_loader(dataloader: Any) -> Any:
    """
    Yield batches from a finite dataloader forever without caching batches.
    """
    while True:
        for batch in dataloader:
            yield batch
