"""
scripts/eval.py
---------------
Full per-perturbation evaluation of a trained VAE checkpoint.

Usage
~~~~~
    # Use the latest checkpoint in the experiment directory
    uv run python scripts/eval.py --train_dir outputs/experiments/my_run

    # Use a specific checkpoint
    uv run python scripts/eval.py --train_dir outputs/experiments/my_run \\
        --checkpoint step_0010000

The script reads ``{train_dir}/config.yaml`` to reconstruct the dataset and
model exactly as they were during training, then evaluates the val split.

Two metric families are reported (the legacy "proxy" per-pert summary — noisy
fast top-k DEG approximations — was removed; those live only as train-time
monitoring in ``train_utils.run_validation``):

cell-eval metrics  (Mann-Whitney U + FDR; the paper's per-perturbation suite)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  discrimination_score_l1   — Wu et al. 2024 inverse-rank, L1 metric (Fig 2D)
  pearson_delta             — Pearson on all-gene Δ vectors          (Fig 2E)
  pr_auc                    — AP w/ FDR<0.05 labels, -log10(p) score (Fig 2F)
  de_spearman_lfc_sig       — Spearman log2FC over real-FDR-sig set (Fig 2H)
  mae_log2fc_sig            — MAE of log2FC over real-FDR-sig set   (scLDM Fig13)
  overlap_at_n              — Top-N FDR-sig intersection/N (N = #real_sig)
                                                                    (Fig 2I)
  de_spearman_sig           — Across-pert Spearman of #sig per pert (Fig 2J)
  mae_pseudobulk            — all-gene pseudobulk MAE                (scLDM Fig13)

scLDM distribution-level metrics  (PCA-30 features, pooled across perts)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  w2_pca30                  — 2-Wasserstein via OT (POT lib)        (Tbl 3)
  mmd2_rbf_pca30            — squared MMD with RBF kernel           (Tbl 3)
  fd_pca30                  — Fréchet Distance (FID-style)          (Tbl 3)

Outputs (in ``{train_dir}/eval_results/`` or ``--output_dir``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  per_pert_metrics.json     — per-perturbation metric dict
  per_pert_metrics.csv      — same, as CSV
  summary_metrics.json      — mean ± std across perturbations
                              (keys: cell_eval + distribution)
  state_per_pert.json       — per-pert cell-eval scores
  real_de_table.csv.gz      — real DE table (target × gene rows)
  pred_de_table.csv.gz      — predicted DE table
"""

import csv
import json
import logging
import os
from collections import defaultdict

import numpy as np
import draccus
import torch
from torch import Tensor
from torch.utils.data import Subset
from tqdm.auto import tqdm

from sc_evae.config.eval import EvalConfig
from sc_evae.config.training import TrainConfig
from sc_evae.metrics.perturbation import _pearson
from sc_evae.metrics.de_table import build_de_table
from sc_evae.metrics.distribution import (
    frechet_distance_pca,
    mmd2_rbf_pca,
    mmd2_rbf_scldm,
    wasserstein2_pca,
)
from sc_evae.metrics.state_metrics import (
    aggregate_per_pert,
    de_spearman_lfc_sig,
    de_spearman_sig,
    discrimination_score_l1,
    mae_log2fc_sig,
    overlap_at_n,
    pr_auc,
)
from sc_evae.metrics.pfizer_ranking import calinski_harabasz_ratio, l2_norm
from sc_evae.models.autoregressive import AutoregressiveModel
from sc_evae.models.common import extract_pert_cond_input
from sc_evae.models.factory import make_model
from sc_evae.models.flow_matching import FlowMatchingModel
from sc_evae.models.masked_diffusion import MaskedDiffusionModel
from sc_evae.models.vae import sc_evae
from sc_evae.training.data_loader import (
    PerturbationDataset,
    make_dataloader_from_dataset,
)
from sc_evae.training.train_utils import (
    build_deg_table_from_dataset,
    latest_checkpoint,
    resolve_train_val_split,
    to_metric_space,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _load_model_weights(
    model: sc_evae, checkpoint_path: str, device: torch.device
) -> None:
    """
    Load model weights from an Accelerate checkpoint directory.

    Accelerate saves either ``model.safetensors`` (preferred) or
    ``pytorch_model.bin``.  Strips the ``module.`` prefix added by DDP.
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

    # Re-create aliases for shared parameters that state_dict() dedups on save
    # (when two parameter names resolve to the same Parameter, only one is
    # written; load_state_dict iterates per-module and expects both names).
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    for missing in model_keys - ckpt_keys:
        for ckpt_key in ckpt_keys:
            if ckpt_key.endswith(missing) or missing.endswith(ckpt_key):
                if state_dict[ckpt_key].shape == model.state_dict()[missing].shape:
                    state_dict[missing] = state_dict[ckpt_key]
                    break

    model.load_state_dict(state_dict)


def _generate_predictions(
    model,
    batch: dict[str, Tensor],
    num_sample_steps: int | None,
) -> Tensor:
    """Return ``(B, G)`` generated expression — one draw per val cell.

    Per-perturbation averaging over N_P val cells (each an iid draw from the
    predictive distribution for that perturbation) is what turns these
    single-cell samples into a per-pert mean downstream.  No per-cell averaging
    is performed here — that would be redundant for Pearson/DEG and would
    artificially shrink MMD.

    - ``sc_evae``: deterministic encode-quantize-decode via ``predict``.
    - ``AutoregressiveModel`` / ``MaskedDiffusionModel`` / ``FlowMatchingModel``:
      real ancestral / diffusion / ODE sampling, one draw per cell.

    Mode A generative priors (decoder on VAE latent codes) have their sample
    outputs decoded through the frozen VAE using the observed per-cell library
    size so values are comparable with ``batch['counts']``.
    """
    inner = model
    counts = batch["counts"]
    B = counts.shape[0]
    device = counts.device
    cls_cond = None

    if isinstance(inner, sc_evae):
        # VAE is deterministic (encode-quantize-decode); pseudo_sample is the
        # generation (reconstruction at the observed library size).
        return inner.pseudo_sample(batch).float()

    # Pull the right pert-cond input for the prior (pert_emb for 'projected'
    # mode → ESM2 features; pert_idx for 'learned' mode → integer index into
    # nn.Embedding). The downstream pert_proj on the model side accepts whichever
    # was wired at training time. The VAE's decoder may be configured
    # independently, so it gets its own extraction below.
    pert_cond_prior = extract_pert_cond_input(batch, inner.config)
    cell_type_idx = batch.get("cell_type_idx")
    if isinstance(inner, AutoregressiveModel):
        s = inner.sample(
            batch_size=B,
            pert_emb=pert_cond_prior,
            temperature=1.0,
            device=device,
            cls_cond=cls_cond,
            cell_type_idx=cell_type_idx if inner.config.do_celltype_cond else None,
        )
        if inner.vae is not None:
            codes = inner.vae.quantizer.decode_indices(s)  # (B, L, d_z)
            pert_emb_dec = extract_pert_cond_input(batch, inner.vae.config)
            cti_dec = cell_type_idx if inner.vae.config.do_celltype_cond else None
            lib_size = counts.sum(-1, keepdim=True)
            s = inner.vae.decode_observed(
                codes, pert_emb_dec, lib_size, cell_type_idx=cti_dec
            )
    elif isinstance(inner, MaskedDiffusionModel):
        s = inner.sample(
            batch_size=B,
            num_steps=num_sample_steps,
            pert_emb=pert_cond_prior,
            device=device,
            cls_cond=cls_cond,
            cell_type_idx=cell_type_idx if inner.config.do_celltype_cond else None,
        )
        if inner.vae is not None:
            codes = inner.vae.quantizer.decode_indices(s)  # (B, L, d_z)
            pert_emb_dec = extract_pert_cond_input(batch, inner.vae.config)
            cti_dec = cell_type_idx if inner.vae.config.do_celltype_cond else None
            lib_size = counts.sum(-1, keepdim=True)
            s = inner.vae.decode_observed(
                codes, pert_emb_dec, lib_size, cell_type_idx=cti_dec
            )
    elif isinstance(inner, FlowMatchingModel):
        s = inner.sample(
            batch_size=B,
            num_steps=num_sample_steps,
            pert_emb=pert_cond_prior,
            device=device,
            cls_cond=cls_cond,
            cell_type_idx=cell_type_idx if inner.config.do_celltype_cond else None,
        )
        if inner.vae is not None:
            pert_emb_dec = extract_pert_cond_input(batch, inner.vae.config)
            cti_dec = cell_type_idx if inner.vae.config.do_celltype_cond else None
            lib_size = counts.sum(-1, keepdim=True)
            s = inner.vae.decode_observed(
                s, pert_emb_dec, lib_size, cell_type_idx=cti_dec
            )
    else:
        raise NotImplementedError(
            f"No eval-time sampling wired for {type(inner).__name__}"
        )
    return s.float()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@draccus.wrap()
def main(cfg: EvalConfig) -> None:
    if not cfg.train_dir:
        raise ValueError("--train_dir must be set")

    # ---- logging ------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    )
    logger = logging.getLogger(__name__)

    # ---- load training config from saved config.yaml ------------------------
    train_config_path = os.path.join(cfg.train_dir, "config.yaml")
    if not os.path.exists(train_config_path):
        raise FileNotFoundError(
            f"Training config not found at {train_config_path!r}. "
            "Make sure --train_dir points to an experiment directory."
        )
    with open(train_config_path) as f:
        train_cfg: TrainConfig = draccus.load(TrainConfig, f)
    logger.info("Loaded training config from %s", train_config_path)

    if cfg.split_path is not None and cfg.split_path != train_cfg.dataset.split_path:
        logger.info(
            "Overriding dataset.split_path: %r → %r (eval-time only; "
            "saved config.yaml unchanged)",
            train_cfg.dataset.split_path,
            cfg.split_path,
        )
        train_cfg.dataset.split_path = cfg.split_path
        train_cfg.dataset.__post_init__()

    # ---- output directory ---------------------------------------------------
    output_dir = cfg.output_dir or os.path.join(cfg.train_dir, "eval_results")
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Results will be written to %s", output_dir)

    # ---- device -------------------------------------------------------------
    device = _resolve_device(cfg.device)
    logger.info("Using device: %s", device)

    # ---- dataset + split (same seed/mode as training) ----------------------
    logger.info("Loading dataset from %s", train_cfg.dataset.h5ad_path)
    ds = PerturbationDataset(config=train_cfg.dataset, logger=logger)
    # TODO: ensure that same seed gives same split across runs
    train_ds, val_ds = resolve_train_val_split(ds, train_cfg.dataset, train_cfg.seed)
    logger.info(
        "Split: %d train / %d val cells across %d perturbations",
        len(train_ds),
        len(val_ds),
        ds.num_perts,
    )

    # ctrl_label resolved up-front so the subsample loop below can use a
    # different cap for the ctrl pert (max_ctrl_cells, typically larger than
    # max_cells_per_pert so per-gene LFC means stay stable on sparse HVGs).
    ctrl_label = cfg.ctrl_label or train_cfg.dataset.ctrl_label

    # ---- balanced per-pert subsample of val_ds -----------------------------
    # Held-out splits can put millions of cells in val (parse: ~2.3M, ~25k
    # cells/pert). Encoding all of them blows host RAM and gives no extra
    # statistical power. Pick a deterministic random subsample per pert,
    # then sort the union so the lazy h5ad still does sequential reads.
    cap = cfg.max_cells_per_pert
    ctrl_cap = cfg.max_ctrl_cells
    if (cap or ctrl_cap) and len(val_ds) > 0:
        val_pert_idxs = (
            ds._pert_indices[val_ds._indices]
            if val_ds._indices is not None
            else val_ds._pert_indices
        )
        # ctrl_pert_idx may be -1 if ctrl_label is not in the dataset's pert
        # vocabulary; in that case ctrl_cap is moot (no ctrl rows to cap).
        ctrl_pert_idx_local = (
            ds.pert_to_idx.get(ctrl_label, -1) if ctrl_label is not None else -1
        )
        rng = np.random.default_rng(train_cfg.seed)
        keep_lists: list[np.ndarray] = []
        unique_perts_in_val = np.unique(val_pert_idxs)
        for p in unique_perts_in_val:
            cells = np.where(val_pert_idxs == p)[0]
            this_cap = ctrl_cap if int(p) == ctrl_pert_idx_local else cap
            if this_cap and len(cells) > this_cap:
                cells = rng.choice(cells, size=this_cap, replace=False)
            keep_lists.append(cells)
        keep_indices = np.sort(np.concatenate(keep_lists))
        if len(keep_indices) < len(val_ds):
            logger.info(
                "Subsampled val_ds: %d → %d cells (cap=%d per non-ctrl pert, "
                "ctrl_cap=%d, across %d val perts)",
                len(val_ds),
                len(keep_indices),
                cap,
                ctrl_cap,
                len(unique_perts_in_val),
            )
            val_ds = Subset(val_ds, keep_indices.tolist())

    # ---- DEG table from full dataset -----------------------------------------
    # Resolution order mirrors scripts/train.py:
    #   1. DatasetConfig.deg_table_path (YAML override)
    #   2. constants.resolve_default_deg_table_path(...) (registered defaults)
    #   3. compute on the fly (last resort)
    # Shape check (mask width vs num_genes, indices len vs top_k) catches a
    # cache built for a different gene panel / top_k.
    from sc_evae.training.constants import resolve_default_deg_table_path

    cached_path = train_cfg.dataset.deg_table_path or resolve_default_deg_table_path(
        train_cfg.dataset.h5ad_path or None,
        list(train_cfg.dataset.h5ad_paths) if train_cfg.dataset.h5ad_paths else None,
    )
    deg_table: dict | None = None
    if cached_path and os.path.exists(cached_path):
        logger.info("Loading DEG table from %s …", cached_path)
        try:
            cached = torch.load(cached_path, weights_only=False)
            sample_entry = next(iter(cached.values()))
            mask_w = sample_entry["mask"].shape[0]
            idx_len = sample_entry["indices"].shape[0]
            if mask_w != ds.num_genes:
                raise ValueError(
                    f"DEG-table mask width {mask_w} != ds.num_genes {ds.num_genes}; "
                    "cached table was built with a different gene panel."
                )
            if idx_len != cfg.top_k_deg:
                raise ValueError(
                    f"DEG-table indices length {idx_len} != top_k {cfg.top_k_deg}; "
                    "cached table was built with a different top_k."
                )
            deg_table = cached
            logger.info(
                "  Loaded %d perts (mask_w=%d, top_k=%d). Skipping build_deg_table_from_dataset.",
                len(deg_table),
                mask_w,
                idx_len,
            )
        except Exception as e:
            logger.warning(
                "Failed to load cached DEG table from %s: %s. Falling back to recompute.",
                cached_path,
                e,
            )
            deg_table = None

    if deg_table is None:
        logger.info(
            "Building DEG table (ctrl_label=%r, top_k=%d) …", ctrl_label, cfg.top_k_deg
        )
        deg_table = build_deg_table_from_dataset(
            ds, train_cfg.dataset, top_k=cfg.top_k_deg
        )
    logger.info(
        "DEG table ready: %d perturbations (covers train + val perts)", len(deg_table)
    )

    # ---- checkpoint ---------------------------------------------------------
    checkpoint_dir = os.path.join(cfg.train_dir, "checkpoints")
    if cfg.checkpoint is not None:
        checkpoint_path = os.path.join(checkpoint_dir, cfg.checkpoint)
        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path!r}")
    else:
        checkpoint_path = latest_checkpoint(checkpoint_dir)
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir!r}")
    logger.info("Loading checkpoint: %s", checkpoint_path)

    # ---- model --------------------------------------------------------------
    model = make_model(train_cfg.model).to(device)
    _load_model_weights(model, checkpoint_path, device)
    model.eval()
    logger.info(
        "Model loaded (%d parameters)", sum(p.numel() for p in model.parameters())
    )

    logger.info(
        "Generative eval: one sample per cell, num_sample_steps=%s (model=%s)",
        cfg.num_sample_steps,
        type(model).__name__,
    )
    logger.info("Metric-space basis: %s", cfg.cp10k_basis)

    has_encode = hasattr(model, "encode")

    # ---- inference: accumulate per-perturbation predictions -----------------
    # shuffle=False so the lazy h5ad does sequential reads. The Subset above
    # already supplies a balanced random per-pert subsample; iterating it in
    # ascending index order is what makes the lazy backend fast.
    val_dl = make_dataloader_from_dataset(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )

    # per_pert_preds[label]   = list of (G,) tensors (predicted mean expression)
    # per_pert_true[label]    = list of (G,) tensors (observed expression)
    # per_pert_latents[label] = list of (D,) tensors (flattened pre-quant latent)
    def _accumulate(dl, desc):
        preds: dict[str, list[Tensor]] = defaultdict(list)
        true: dict[str, list[Tensor]] = defaultdict(list)
        latents: dict[str, list[Tensor]] = defaultdict(list)
        with torch.no_grad():
            for batch in tqdm(dl, desc=desc):
                batch_device = {
                    k: v.to(device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                }
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = _generate_predictions(
                        model,
                        batch_device,
                        num_sample_steps=cfg.num_sample_steps,
                    )  # (B, G)
                pred = pred.float()

                if has_encode:
                    latent = model.encode(batch_device["counts"].float())  # (B, L, C)
                    latent_flat = latent.reshape(latent.shape[0], -1).cpu()
                else:
                    latent_flat = None

                # Convert to log1p(CP10K) space for all downstream metrics.
                # Pass both library_size (full pre-HVG, from data loader) and
                # hvg_library (HVG row sum from batch["counts"]) so to_metric_space
                # can produce CP10K under either basis (cfg.cp10k_basis):
                #   - "full_library"  → absolute CP10K, scVI/scanpy convention
                #   - "hvg_row_sum"   → HVG-fractional CP10K, scLDM/State paper
                # The hvg_row_sum basis applies a per-cell full_lib/HVG_lib
                # rescale in the log1p-baked branch so all model variants
                # (log1p-trained, raw-counts-trained) sit on the same scale.
                lib = batch.get("library_size")
                lib_cpu = lib.cpu() if lib is not None else None
                hvg_lib_cpu = batch["counts"].float().sum(dim=-1, keepdim=True).cpu()
                pred_metric = to_metric_space(
                    pred.cpu(),
                    train_cfg.dataset,
                    library_size=lib_cpu,
                    hvg_library=hvg_lib_cpu,
                    basis=cfg.cp10k_basis,
                )
                counts_metric = to_metric_space(
                    batch["counts"].float(),
                    train_cfg.dataset,
                    library_size=lib_cpu,
                    hvg_library=hvg_lib_cpu,
                    basis=cfg.cp10k_basis,
                )

                for i, pert_idx in enumerate(batch["pert_idx"]):
                    label = ds.unique_perts[pert_idx.item()]
                    preds[label].append(pred_metric[i])
                    true[label].append(counts_metric[i])
                    if latent_flat is not None:
                        latents[label].append(latent_flat[i])
        return preds, true, latents

    per_pert_preds, per_pert_true, per_pert_latents = _accumulate(
        val_dl, "encoding val set"
    )

    # ---- per-perturbation metrics -------------------------------------------
    per_pert_results: dict[str, dict] = {}
    skipped_no_table = 0
    ctrl_cells_skipped = 0

    # ctrl cells from the val set are kept aside — they are the reference for
    # the State / cell-eval DE tables (matches cell-eval's ``mode='ref'``).
    ctrl_pred_cells: list[Tensor] = per_pert_preds.pop(ctrl_label, [])
    ctrl_true_cells: list[Tensor] = per_pert_true.pop(ctrl_label, [])
    ctrl_cells_skipped = len(ctrl_true_cells)

    # Some published splits (e.g. scLDM HepG2 cell-type × pert holdout) put
    # the held-out perts in val but leave ctrl cells of the held-out cell
    # type in train. When val has no ctrl, pull a matched ctrl reference
    # from the same cell type(s) elsewhere in `ds` and run inference on it
    # so the State / cell-eval DE block can still build its tables.
    if not ctrl_true_cells and ctrl_label in ds.pert_to_idx:
        if isinstance(val_ds, Subset):
            val_orig_idx = val_ds.dataset._indices[np.asarray(val_ds.indices)]
        else:
            val_orig_idx = val_ds._indices
        val_cell_types_idx = np.unique(ds._cell_type_indices[val_orig_idx])
        ctrl_pert_idx = ds.pert_to_idx[ctrl_label]
        ctrl_mask = (ds._pert_indices == ctrl_pert_idx) & np.isin(
            ds._cell_type_indices, val_cell_types_idx
        )
        ctrl_orig_idx = np.where(ctrl_mask)[0]
        if ctrl_orig_idx.size == 0:
            logger.warning(
                "Train fallback for ctrl reference: no ctrl cells (%r) found "
                "in val cell type(s) %s — skipping State / cell-eval metrics.",
                ctrl_label,
                [ds.unique_cell_types[i] for i in val_cell_types_idx],
            )
        else:
            ctrl_cap = cfg.max_ctrl_cells
            if ctrl_cap and len(ctrl_orig_idx) > ctrl_cap:
                rng_ctrl = np.random.default_rng(train_cfg.seed)
                ctrl_orig_idx = np.sort(
                    rng_ctrl.choice(ctrl_orig_idx, size=ctrl_cap, replace=False)
                )
            logger.info(
                "Val split has no %r cells; using %d ctrl cells from train "
                "(cell types: %s) as the DE reference (scLDM convention).",
                ctrl_label,
                len(ctrl_orig_idx),
                [ds.unique_cell_types[i] for i in val_cell_types_idx],
            )
            ctrl_ds = Subset(ds, ctrl_orig_idx.tolist())
            ctrl_dl = make_dataloader_from_dataset(
                ctrl_ds,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                shuffle=False,
            )
            ctrl_preds_dict, ctrl_true_dict, _ = _accumulate(
                ctrl_dl, "sampling ctrl ref from train"
            )
            ctrl_pred_cells = ctrl_preds_dict.get(ctrl_label, [])
            ctrl_true_cells = ctrl_true_dict.get(ctrl_label, [])
            ctrl_cells_skipped = len(ctrl_true_cells)

    # Control reference means (log1p-CP10K space) for the cell-eval pearson_delta.
    # ``ctrl_*_cells`` were restricted to the val (held-out) cell type(s) above,
    # so these never mix in control cells from other cell lines — critical when
    # the training data spans multiple cell lines (e.g. Replogle hepg2 holdout
    # vs. k562/jurkat/rpe1). Each side uses its OWN control: real Δ vs the real
    # control mean, predicted Δ vs the *predicted* control mean — matching
    # ``cell_eval.metrics.pearson_delta`` / ``perturbation_effect``.
    true_ctrl_mean = torch.stack(ctrl_true_cells).mean(0) if ctrl_true_cells else None
    pred_ctrl_mean = (
        torch.stack(ctrl_pred_cells).mean(0) if ctrl_pred_cells else true_ctrl_mean
    )

    for label, pred_list in sorted(per_pert_preds.items()):
        if label not in deg_table:
            skipped_no_table += 1
            logger.debug("Perturbation %r not in DEG table — skipping.", label)
            continue

        pred_cells = torch.stack(pred_list)  # (N, G)
        true_cells = torch.stack(per_pert_true[label])  # (N, G)
        pred_mean = pred_cells.mean(0)  # (G,)
        true_mean = true_cells.mean(0)  # (G,)

        # pearson_delta: corr(pred − pred_ctrl, real − real_ctrl) over ALL genes
        # (STATE Fig 2E). mae_pseudobulk: all-gene pseudobulk MAE (scLDM Fig 13).
        # The former noisy proxy_* family (pearson_deg20, r2_deg20, mmd,
        # deg_auprc, …) was dropped — eval reports only the cell-eval +
        # distribution numbers below.
        if true_ctrl_mean is not None:
            pearson_delta = _pearson(
                pred_mean - pred_ctrl_mean, true_mean - true_ctrl_mean
            )
        else:
            pearson_delta = float("nan")
        per_pert_results[label] = {
            "n_cells": len(pred_list),
            "pearson_delta": pearson_delta,
            "mae_pseudobulk": float(
                (pred_mean.float() - true_mean.float()).abs().mean().item()
            ),
        }

    logger.info(
        "Evaluated %d perturbations  (skipped %d without DEG table entry, "
        "%d control batches)",
        len(per_pert_results),
        skipped_no_table,
        ctrl_cells_skipped,
    )

    # ---- State / cell-eval-faithful metrics --------------------------------
    state_per_pert: dict[str, dict[str, float]] = {}
    state_summary: dict[str, dict] = {}
    if not ctrl_true_cells:
        logger.warning(
            "No ctrl cells (%r) present in val split — skipping State / "
            "cell-eval metrics. (They require ctrl as the DE reference.)",
            ctrl_label,
        )
    else:
        gene_names: list[str] | None = getattr(ds, "_target_genes", None)
        if gene_names is None:
            try:
                import anndata

                _adata = anndata.read_h5ad(train_cfg.dataset.h5ad_path, backed="r")
                gene_names = _adata.var_names.to_numpy().tolist()
                _adata.file.close()
            except Exception as e:  # pragma: no cover
                logger.warning("Could not resolve gene names (%s); using indices.", e)
                gene_names = [str(i) for i in range(ds.num_genes)]
        if len(gene_names) > ds.num_genes:
            gene_names = gene_names[: ds.num_genes]

        true_ctrl = torch.stack(ctrl_true_cells)  # (N_c, G)
        pred_ctrl = torch.stack(ctrl_pred_cells) if ctrl_pred_cells else true_ctrl
        # Materialise per-pert (N_p, G) tensors for both real and predicted.
        real_per_pert = {
            label: torch.stack(per_pert_true[label]) for label in per_pert_results
        }
        pred_per_pert = {
            label: torch.stack(per_pert_preds[label]) for label in per_pert_results
        }

        # ---- real DE table (cache-aware) ----
        # Mann-Whitney + BH-FDR over the full gene panel × all val perts is
        # the single most expensive CPU step in eval.py. It depends only on
        # (val cells, ctrl cells, gene names, de_epsilon) — not on the model —
        # so we cache it. Resolution: explicit cfg.de_table_path → registered
        # default for (h5ad, split-spec stem, ε) → recompute. On miss, write the
        # rebuilt table back to the resolved path so parallel checkpoints
        # share the artifact.
        from sc_evae.training.constants import resolve_default_de_table_path
        import pandas as pd

        de_cache_path = cfg.de_table_path or resolve_default_de_table_path(
            train_cfg.dataset.h5ad_path or None,
            (
                list(train_cfg.dataset.h5ad_paths)
                if train_cfg.dataset.h5ad_paths
                else None
            ),
            (
                cfg.split_path
                if cfg.split_path is not None
                else train_cfg.dataset.split_path
            ),
            cfg.de_epsilon,
        )
        real_de = None
        if de_cache_path and os.path.exists(de_cache_path):
            logger.info("Loading cached real_de from %s …", de_cache_path)
            try:
                cached = pd.read_csv(de_cache_path, compression="gzip")
                cached_genes = set(cached["gene"].astype(str).unique())
                cached_perts = set(cached["target"].astype(str).unique())
                want_genes = set(map(str, gene_names))
                want_perts = set(map(str, real_per_pert))
                if not want_genes.issubset(cached_genes):
                    raise ValueError(
                        f"cached gene set ({len(cached_genes)} genes) does not "
                        f"cover current gene_names ({len(want_genes)} genes); "
                        f"missing {len(want_genes - cached_genes)} (e.g. "
                        f"{sorted(want_genes - cached_genes)[:3]})."
                    )
                missing_perts = want_perts - cached_perts
                if missing_perts:
                    raise ValueError(
                        f"cached real_de does not cover {len(missing_perts)} val "
                        f"perts (e.g. {sorted(missing_perts)[:5]}); cache built "
                        "for a different split."
                    )
                real_de = cached
                logger.info(
                    "  Loaded %d rows (%d perts × %d genes). Skipping rebuild.",
                    len(real_de),
                    len(cached_perts),
                    len(cached_genes),
                )
            except Exception as e:
                logger.warning(
                    "Failed to load cached real_de from %s: %s. Rebuilding.",
                    de_cache_path,
                    e,
                )
                real_de = None

        if real_de is None:
            logger.info(
                "Building real DE table over %d perts (ctrl=%d cells, de_epsilon=%s) …",
                len(real_per_pert),
                true_ctrl.shape[0],
                cfg.de_epsilon,
            )
            real_de = build_de_table(
                real_per_pert,
                true_ctrl,
                gene_names,
                epsilon=cfg.de_epsilon,
            )
            if de_cache_path:
                try:
                    os.makedirs(os.path.dirname(de_cache_path) or ".", exist_ok=True)
                    # Atomic write so concurrent RunPod evals don't see a
                    # half-written file: write to pid-suffixed temp + rename.
                    tmp_path = f"{de_cache_path}.tmp.{os.getpid()}"
                    real_de.to_csv(tmp_path, index=False, compression="gzip")
                    os.replace(tmp_path, de_cache_path)
                    logger.info("Saved real_de cache: %s", de_cache_path)
                except Exception as e:
                    logger.warning(
                        "Failed to save real_de cache to %s: %s "
                        "(continuing without caching).",
                        de_cache_path,
                        e,
                    )

        logger.info(
            "Building predicted DE table over %d perts (ctrl=%d cells, de_epsilon=%s) …",
            len(pred_per_pert),
            pred_ctrl.shape[0],
            cfg.de_epsilon,
        )
        pred_de = build_de_table(
            pred_per_pert,
            pred_ctrl,
            gene_names,
            epsilon=cfg.de_epsilon,
        )

        # Per-pert pseudobulks for the discrimination score (log1p space).
        # cell-eval / STATE compute the discrimination score on the arithmetic
        # mean of the log1p .X — BulkArrays._bulk_anndata does
        # group_by().mean() on the (already log-normed) .X, with NO expm1.
        # STATE feeds log1p pseudobulks (clipped to [0, 14]) straight into
        # MetricsEvaluator, whose _convert_to_normlog leaves log-norm data
        # untouched (state/.../_predict.py:532-537,584). So we mean the log1p
        # cells directly — feeding expm1(count) here would put PDS in a
        # different space than their reported numbers.
        labels_sorted = sorted(real_per_pert)
        real_pb = np.stack(
            [real_per_pert[k].numpy().mean(axis=0) for k in labels_sorted]
        )
        pred_pb = np.stack(
            [pred_per_pert[k].numpy().mean(axis=0) for k in labels_sorted]
        )
        ctrl_pb_real = true_ctrl.numpy().mean(axis=0)

        disc_l1 = discrimination_score_l1(
            real_pb,
            pred_pb,
            ctrl_pb_real,
            labels_sorted,
            metric="l1",
            exclude_target_gene=True,
            gene_names=gene_names,
        )
        ovl = overlap_at_n(real_de, pred_de, n=None, fdr_threshold=0.05)
        spr_lfc = de_spearman_lfc_sig(real_de, pred_de, fdr_threshold=0.05)
        spr_sig = de_spearman_sig(real_de, pred_de, fdr_threshold=0.05)
        prauc = pr_auc(real_de, pred_de, fdr_threshold=0.05)
        mae_lfc = mae_log2fc_sig(real_de, pred_de, fdr_threshold=0.05)

        # cell-eval's pearson_delta is the all-gene (no DEG mask) Δ correlation
        # matching STATE Fig 2E "Pearson Correlation (Delta)"; mae_pseudobulk is
        # the cell-eval / scLDM Fig 13 all-gene pseudobulk MAE column. Both were
        # computed per-pert in the loop above.
        pearson_delta = {
            label: per_pert_results[label]["pearson_delta"] for label in labels_sorted
        }
        mae_pb = {
            label: per_pert_results[label]["mae_pseudobulk"] for label in labels_sorted
        }

        for label in labels_sorted:
            state_per_pert[label] = {
                "discrimination_score_l1": disc_l1.get(label, float("nan")),
                "overlap_at_n": ovl.get(label, float("nan")),
                "de_spearman_lfc_sig": spr_lfc.get(label, float("nan")),
                "mae_log2fc_sig": mae_lfc.get(label, float("nan")),
                "pr_auc": prauc.get(label, float("nan")),
                "pearson_delta": pearson_delta.get(label, float("nan")),
                "mae_pseudobulk": mae_pb.get(label, float("nan")),
            }

        for name, scores in [
            ("discrimination_score_l1", disc_l1),
            ("overlap_at_n", ovl),
            ("de_spearman_lfc_sig", spr_lfc),
            ("mae_log2fc_sig", mae_lfc),
            ("pr_auc", prauc),
            ("pearson_delta", pearson_delta),
            ("mae_pseudobulk", mae_pb),
        ]:
            state_summary[name] = aggregate_per_pert(scores)
        # de_spearman_sig is a single across-pert scalar — no aggregation.
        state_summary["de_spearman_sig"] = {
            "value": float(spr_sig),
            "n_perts": len(labels_sorted),
        }

        logger.info(
            "Summary metrics — State / cell-eval (mean ± std across %d perturbations):",
            len(labels_sorted),
        )
        for name in [
            "discrimination_score_l1",
            "overlap_at_n",
            "de_spearman_lfc_sig",
            "mae_log2fc_sig",
            "pr_auc",
            "pearson_delta",
            "mae_pseudobulk",
        ]:
            stats = state_summary[name]
            logger.info(
                "  %-26s  %.4f ± %.4f  (n=%d)",
                name,
                stats["mean"],
                stats["std"],
                stats["n"],
            )
        logger.info(
            "  %-26s  %.4f  (across %d perts)",
            "de_spearman_sig",
            state_summary["de_spearman_sig"]["value"],
            state_summary["de_spearman_sig"]["n_perts"],
        )

    # ---- scLDM distribution-level metrics (W2, MMD2 RBF, FD on PCA-30) -----
    # Pool predictions and observations across all evaluated perturbations
    # (excluding ctrl) into two big (N, G) tensors; PCA-fit on the pooled
    # true cells; compute three scalars on PCA features. Mirrors scLDM
    # Table 3 reporting on Replogle / Parse1M.
    distribution_summary: dict[str, float] = {}
    if per_pert_results:
        pred_pool_list: list[Tensor] = []
        true_pool_list: list[Tensor] = []
        for label in per_pert_results:
            pred_pool_list.extend(per_pert_preds[label])
            true_pool_list.extend(per_pert_true[label])
        pred_pool = torch.stack(pred_pool_list).float().numpy()
        true_pool = torch.stack(true_pool_list).float().numpy()
        logger.info(
            "Distribution metrics: pooled %d pred / %d true cells across %d perts",
            pred_pool.shape[0],
            true_pool.shape[0],
            len(per_pert_results),
        )
        distribution_summary["w2_pca30"] = wasserstein2_pca(
            pred_pool,
            true_pool,
            n_components=cfg.pca_components,
            method=cfg.w2_method,
            max_samples=cfg.distribution_max_samples,
            seed=0,
        )
        distribution_summary["mmd2_rbf_pca30"] = mmd2_rbf_pca(
            pred_pool,
            true_pool,
            n_components=cfg.pca_components,
            max_samples=cfg.distribution_max_samples,
            seed=0,
        )
        distribution_summary["fd_pca30"] = frechet_distance_pca(
            pred_pool,
            true_pool,
            n_components=cfg.pca_components,
            max_samples=cfg.distribution_max_samples,
            seed=0,
        )
        # scLDM Table-3-style MMD: mixture of RBF kernels (σ ∈ {1,2,4,8,16}),
        # biased estimator, PCA-30 fit on true. Mirrors
        # scldm.mmd.mix_rbf_mmd2 (the multi-kernel recipe — Table 3 reports
        # MMD² > 2 for some baselines, which exceeds the single-kernel max).
        distribution_summary["mmd2_rbf_scldm"] = mmd2_rbf_scldm(
            pred_pool,
            true_pool,
            n_components=cfg.pca_components,
            max_samples=cfg.distribution_max_samples,
            seed=0,
        )
        # Per-gene first/second moment R² across the pooled cells
        # (scLDM models.py R2_METRICS).
        from sklearn.metrics import r2_score as _r2

        distribution_summary["r2_mean"] = float(
            _r2(true_pool.mean(axis=0), pred_pool.mean(axis=0))
        )
        distribution_summary["r2_var"] = float(
            _r2(true_pool.var(axis=0), pred_pool.var(axis=0))
        )
        # Per-cell reconstruction PCC / MSE — flattened across all
        # (cell, gene) pairs in the pooled set. Directly comparable to
        # scLDM's torchmetrics.pearson_corrcoef logged in shared_step.
        pred_flat = pred_pool.reshape(-1).astype(np.float64)
        true_flat = true_pool.reshape(-1).astype(np.float64)
        if pred_flat.std() > 0 and true_flat.std() > 0:
            distribution_summary["pcc_per_cell"] = float(
                np.corrcoef(pred_flat, true_flat)[0, 1]
            )
        else:
            distribution_summary["pcc_per_cell"] = float("nan")
        distribution_summary["mse_per_cell"] = float(
            ((pred_flat - true_flat) ** 2).mean()
        )
        logger.info("Distribution metrics — global pool (%d PCs):", cfg.pca_components)
        for name, val in distribution_summary.items():
            logger.info("  %-22s  %.4f", name, val)

    # ---- representation metrics (latent space) ----------------------------------
    # Build one latent matrix + label arrays from val cells.
    # Uses the pre-quantization continuous latent (encoder output before VQ).
    all_latents_list: list[Tensor] = []
    all_labels_rep: list[str] = []
    for label, lat_list in per_pert_latents.items():
        all_latents_list.extend(lat_list)
        all_labels_rep.extend([label] * len(lat_list))

    nan_ch = {
        "true": float("nan"),
        "random_mean": float("nan"),
        "random_std": float("nan"),
        "ratio": float("nan"),
    }

    if not all_latents_list:
        logger.info("Model has no encode() — skipping representation metrics.")
        rep_metrics = {"pert_label_ch": nan_ch, "ctrl_vs_pert_ch": nan_ch}
    else:
        latent_matrix: np.ndarray = l2_norm(
            torch.stack(all_latents_list).numpy()
        )  # (N, D), L2-normalised — same convention as Pfizer eval

        # 1. Perturbation-label CH ratio (non-ctrl cells only).
        #    Measures whether the latent clusters cells by perturbation identity.
        non_ctrl_mask = np.array([l != ctrl_label for l in all_labels_rep])
        non_ctrl_labels = [l for l, m in zip(all_labels_rep, non_ctrl_mask) if m]
        if non_ctrl_mask.sum() >= 2 and len(set(non_ctrl_labels)) >= 2:
            pert_ch = calinski_harabasz_ratio(
                latent_matrix[non_ctrl_mask],
                non_ctrl_labels,
                n_random_trials=cfg.n_ch_random_trials,
            )
        else:
            pert_ch = dict(nan_ch)

        # 2. Control vs. perturbed CH ratio (all val cells, binary label).
        #    Measures whether ctrl cells form a distinct cluster from perturbed cells.
        binary_labels = ["ctrl" if l == ctrl_label else "pert" for l in all_labels_rep]
        if len(set(binary_labels)) >= 2:
            ctrl_pert_ch = calinski_harabasz_ratio(
                latent_matrix,
                binary_labels,
                n_random_trials=cfg.n_ch_random_trials,
            )
        else:
            ctrl_pert_ch = dict(nan_ch)

        rep_metrics = {
            "pert_label_ch": pert_ch,
            "ctrl_vs_pert_ch": ctrl_pert_ch,
        }
        logger.info("Representation metrics (latent space, L2-normalised pre-quant):")
        logger.info(
            "  pert_label CH ratio:    %.3f  (true=%.1f, rand=%.1f)",
            pert_ch["ratio"],
            pert_ch["true"],
            pert_ch["random_mean"],
        )
        logger.info(
            "  ctrl_vs_pert CH ratio:  %.3f  (true=%.1f, rand=%.1f)",
            ctrl_pert_ch["ratio"],
            ctrl_pert_ch["true"],
            ctrl_pert_ch["random_mean"],
        )

    # ---- save results -------------------------------------------------------
    # JSON: per-perturbation
    json_path = os.path.join(output_dir, "per_pert_metrics.json")
    with open(json_path, "w") as f:
        json.dump(per_pert_results, f, indent=2)
    logger.info("Saved: %s", json_path)

    # CSV: per-perturbation
    csv_path = os.path.join(output_dir, "per_pert_metrics.csv")
    if per_pert_results:
        fieldnames = ["perturbation", "n_cells", "pearson_delta", "mae_pseudobulk"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for label, row in sorted(per_pert_results.items()):
                writer.writerow({"perturbation": label, **row})
        logger.info("Saved: %s", csv_path)

    # JSON: summary (cell-eval + distribution — the reported numbers)
    summary_path = os.path.join(output_dir, "summary_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "cell_eval": state_summary,
                "distribution": distribution_summary,
            },
            f,
            indent=2,
        )
    logger.info("Saved: %s", summary_path)

    # State per-pert + DE tables
    if state_per_pert:
        state_path = os.path.join(output_dir, "state_per_pert.json")
        with open(state_path, "w") as f:
            json.dump(state_per_pert, f, indent=2)
        logger.info("Saved: %s", state_path)

        real_de_path = os.path.join(output_dir, "real_de_table.csv.gz")
        pred_de_path = os.path.join(output_dir, "pred_de_table.csv.gz")
        real_de.to_csv(real_de_path, index=False, compression="gzip")
        pred_de.to_csv(pred_de_path, index=False, compression="gzip")
        logger.info("Saved: %s", real_de_path)
        logger.info("Saved: %s", pred_de_path)

    # JSON: representation metrics
    rep_path = os.path.join(output_dir, "representation_metrics.json")
    with open(rep_path, "w") as f:
        json.dump(rep_metrics, f, indent=2)
    logger.info("Saved: %s", rep_path)

    logger.info("Evaluation complete.")

    # ---- W&B artifact upload (attach to the existing training run) ----------
    # Look up the training run by display name in train_cfg.wandb.entity/project,
    # resume it with wandb.init(id=..., resume="must"), and log eval_results as
    # a versioned artifact. No new metrics are logged → no run-list spam, the
    # artifact lives under the training run's "Artifacts" tab. Silent no-op if
    # the run can't be located.
    if (
        cfg.wandb_upload
        and getattr(train_cfg, "wandb", None)
        and train_cfg.wandb.enabled
    ):
        try:
            import wandb

            entity = train_cfg.wandb.entity
            project = train_cfg.wandb.project
            run_name = train_cfg.run_name  # base name, no datetime prefix
            api = wandb.Api()
            matches = list(
                api.runs(
                    f"{entity}/{project}",
                    filters={"display_name": run_name},
                )
            )
            if not matches:
                logger.warning(
                    "W&B upload: no run named %r in %s/%s — skipping artifact.",
                    run_name,
                    entity,
                    project,
                )
            else:
                # If multiple matches, pick the most recent (could be re-runs).
                matches.sort(key=lambda r: r.created_at, reverse=True)
                run_id = matches[0].id
                logger.info(
                    "W&B upload: attaching eval_results to run %s/%s/%s (id=%s)",
                    entity,
                    project,
                    run_name,
                    run_id,
                )
                wandb.init(
                    project=project,
                    entity=entity,
                    id=run_id,
                    resume="must",
                    job_type="eval",
                )
                artifact = wandb.Artifact(
                    name=f"eval_results-{run_name}",
                    type="eval_results",
                    description=(
                        f"scripts/eval.py output for {run_name} "
                        f"(checkpoint={os.path.basename(checkpoint_path)})"
                    ),
                )
                artifact.add_dir(output_dir)
                wandb.log_artifact(artifact)
                wandb.finish()
                logger.info("W&B upload: artifact logged.")
        except Exception as e:  # pragma: no cover
            logger.warning("W&B upload failed: %s (continuing).", e)


if __name__ == "__main__":
    main()
