"""
scripts/eval_pfizer.py
-----------------------
Out-of-distribution embedding quality evaluation on the Pfizer IMRU dataset.

The Pfizer dataset is a CRISPRi knockdown screen under IL-1β + TNFα
inflammation. Its raw ~38k-gene counts are sliced to the model's gene panel
(the scLDM 2,000-gene Replogle panel) on the fly at load time via
``gene_order_path`` — genes missing from the Pfizer file are zero-filled.
We evaluate whether the sc_evae VAE's latent space — trained on Replogle —
organises Pfizer cells in a biologically meaningful way.

Usage
~~~~~
    # The raw Pfizer file is sliced to the model's 2000-gene panel on the fly
    # by the data loader (gene_order_path permutes + zero-fills; num_hvg slices).

    # sc_evae only (smoke test)
    uv run python scripts/eval_pfizer.py \\
        --train_dir outputs/experiments/p1-mse \\
        --h5ad_path datasets/pfizer/pfizer_raw.h5ad \\
        --gene_order_path assets/replogle_scldm2k_gene_order.txt --num_hvg 2000 \\
        --max_cells 5000 --n_random_trials 3

    # full run, no baselines
    uv run python scripts/eval_pfizer.py \\
        --train_dir outputs/experiments/p1-mse \\
        --h5ad_path datasets/pfizer/pfizer_raw.h5ad \\
        --gene_order_path assets/replogle_scldm2k_gene_order.txt --num_hvg 2000

    # with Pfizer foundation-model baselines + compbio_direct comparison
    uv run python scripts/eval_pfizer.py \\
        --train_dir outputs/experiments/p1-mse \\
        --h5ad_path datasets/pfizer/pfizer_raw.h5ad \\
        --gene_order_path assets/replogle_scldm2k_gene_order.txt --num_hvg 2000 \\
        --baseline_dir datasets/pfizer/foundation_model_h5ads/ \\
        --compbio_ranking_path datasets/pfizer/20251203_perturbseq_ranked_list_direct_method_trad.txt

Metrics computed (per method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  1. Phenotypic reversion ranking
       Cosine distance from each Treated-perturbation centroid to the
       Untreated SAFE_TARGET anchor. Lower distance = better reversion candidate.

  2. Treatment-condition Calinski-Harabasz ratio
       CH score (true condition labels) / mean CH score (shuffled labels),
       evaluated on control cells only. Measures disease-vs-healthy separation.

  3. Gene-target Calinski-Harabasz ratio
       Same ratio but with gene_target labels, evaluated separately for Treated
       and Untreated cells. Measures perturbation separability.

  4. Enrichment AUC for positive controls
       AUC of the cumulative-recovery curve for 6 known anti-inflammatory NFκB
       genes (TNFRSF1A, TRADD, JUNB, JUND, NFKB1, NFKB2) in the reversion
       ranking. ~0.5 = random; 1.0 = all positives at top.
       Optionally: Spearman ρ and overlap@k vs compbio_direct reference ranking.

Outputs (in ``{train_dir}/eval_pfizer/`` or ``--output_dir``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ranking_results.csv      — per-method ranked gene targets + cosine distances
  clustering_metrics.json  — CH ratios per method
  enrichment_auc.json      — enrichment AUC per method
  ranking_agreement.json   — Spearman / overlap vs compbio_direct (if provided)
  summary.json             — all scalar metrics per method in one dict
"""

import csv
import glob
import json
import logging
import os
import random

import matplotlib

matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt

import anndata as ad
import draccus
import numpy as np
import torch
from scipy.sparse import issparse
from tqdm.auto import tqdm

from sc_evae.config.eval_pfizer import PfizerEvalConfig
from sc_evae.config.training import TrainConfig
from sc_evae.metrics.pfizer_ranking import (
    calinski_harabasz_ratio,
    centroid_cosine_ranking,
    enrichment_auc,
    l2_norm,
    ranking_agreement,
)
from sc_evae.models.factory import load_weights_from_checkpoint, make_model
from sc_evae.models.vae import sc_evae
from sc_evae.training.data_loader import (
    PerturbationDataset,
    make_dataloader_from_dataset,
)
from sc_evae.training.train_utils import latest_checkpoint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _stratified_subsample(
    cell_ids: list[str],
    group_labels: list[str],
    max_cells: int,
    seed: int = 42,
) -> list[int]:
    """Return indices of a stratified subsample (proportional per group)."""
    rng = random.Random(seed)
    from collections import defaultdict

    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(group_labels):
        group_to_indices[g].append(i)

    total = len(cell_ids)
    selected: list[int] = []
    for g, idxs in group_to_indices.items():
        n_take = max(1, round(len(idxs) / total * max_cells))
        selected.extend(rng.sample(idxs, min(n_take, len(idxs))))

    # Trim to exactly max_cells if we overshot due to rounding
    rng.shuffle(selected)
    return selected[:max_cells]


def _densify(X) -> np.ndarray:
    """Convert sparse or dense matrix to float32 numpy array."""
    if issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def _encode_cells(
    model: sc_evae,
    train_dataset_cfg,  # cloned + path-overridden DatasetConfig
    selected_indices: list[int] | None,  # None = all rows in order
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_post_quantized: bool,
    use_pre_quantproj: bool,
    logger: logging.Logger,
) -> np.ndarray:
    """Encode Pfizer cells in batches; return latent embeddings.

    Routes through ``PerturbationDataset`` so the full preprocessing chain —
    ``apply_normalize`` / ``apply_log1p`` / ``use_binning`` plus, critically,
    ``gene_order_path`` permutation and ``num_hvg`` slicing — runs exactly as
    it did during training. Without this, a model trained on (say) 2 000
    HVGs in CellxGene-HVG order would receive Pfizer's full ~36 k-gene panel
    in Pfizer order and produce shape errors / garbage.

    Output shape depends on settings:
      - use_pre_quantproj=True : (N, d_model) mean-pooled encoder output, e.g. 256
      - use_post_quantized=True: (N, num_latents * codebook_dim) post-quantized codes
      - default                : (N, num_latents * codebook_dim) pre-quantized continuous

    ``selected_indices`` are interpreted as **post-filter** dataset indices.
    Since we override ``min_genes=0`` / ``min_cells=0`` and the Pfizer DATASET_METADATA
    default ``obs_filters: {condition: []}`` is a no-op, dataset indices align
    with raw-adata row indices in practice.
    """
    model.eval()
    embeddings: list[np.ndarray] = []

    indices = (
        np.asarray(selected_indices, dtype=np.int64)
        if selected_indices is not None
        else None
    )
    ds = PerturbationDataset(train_dataset_cfg, indices=indices, logger=logger)
    logger.info(
        "  PerturbationDataset built: %d cells × %d genes  "
        "(gene_order=%s, num_hvg=%s, normalize=%s, log1p=%s, binning=%s)",
        ds.num_cells,
        ds.num_genes,
        train_dataset_cfg.gene_order_path,
        train_dataset_cfg.num_hvg,
        train_dataset_cfg.apply_normalize,
        train_dataset_cfg.apply_log1p,
        train_dataset_cfg.use_binning,
    )
    dl = make_dataloader_from_dataset(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    with torch.no_grad():
        for batch in tqdm(dl, desc="encoding (sc_evae)"):
            counts = batch["counts"].to(device)

            if use_pre_quantproj:
                latent = model._run_encoder(counts).mean(dim=1)  # (B, d_model)
            elif use_post_quantized:
                pre_quant = model.encode(counts)
                latent = model.quantizer(pre_quant).codes
            else:
                latent = model.encode(counts)

            embeddings.append(latent.reshape(latent.shape[0], -1).cpu().numpy())

    result = np.concatenate(embeddings, axis=0)
    logger.info(
        "sc_evae embedding shape: %s  (use_pre_quantproj=%s, use_post_quantized=%s)",
        result.shape,
        use_pre_quantproj,
        use_post_quantized,
    )
    return result


def _run_metrics(
    method: str,
    embeddings: np.ndarray,
    group_labels: list[str],
    condition_labels: list[str],
    gene_target_labels: list[str],
    ctrl_label_set: set[str],
    cfg: PfizerEvalConfig,
    logger: logging.Logger,
) -> dict:
    """Compute all 4 metrics for one embedding matrix. Returns a result dict."""
    emb = l2_norm(embeddings)

    # -- Metric 1: phenotypic reversion ranking (Treated → anchor) ------------
    logger.info("[%s] Computing reversion ranking ...", method)
    ranking_raw = centroid_cosine_ranking(
        emb,
        group_labels,
        candidate_prefix=f"{cfg.candidate_condition}_",
        anchor_label=cfg.anchor_state,
    )
    ranking = [(g, d) for g, d in ranking_raw if g not in ctrl_label_set]
    top25 = ranking[:25]
    logger.info(
        "[%s] Top-25 reversion candidates:\n%s",
        method,
        "\n".join(
            f"  {i+1:3d}. {g:<20s}  dist={d:.4f}" for i, (g, d) in enumerate(top25)
        ),
    )

    # -- Metric 1b: untreated ranking (Untreated → same anchor) ---------------
    #    d_untreated(X) = cosine distance of Untreated_X centroid to anchor
    logger.info("[%s] Computing untreated-arm ranking ...", method)
    untreated_ranking_raw = centroid_cosine_ranking(
        emb,
        group_labels,
        candidate_prefix="Untreated_",
        anchor_label=cfg.anchor_state,
    )
    untreated_ranking = [
        (g, d) for g, d in untreated_ranking_raw if g not in ctrl_label_set
    ]

    # -- Selectivity: d_untreated(X) - d_treated(X) ---------------------------
    #    Measures condition-specificity: a positive value means the treated-arm
    #    centroid is closer to the healthy anchor than the untreated-arm centroid,
    #    i.e. the perturbation's effect on cell state is larger under inflammation
    #    than in healthy cells.  High values indicate condition-dependent action
    #    but do NOT guarantee that untreated cells are unperturbed (d_u could be
    #    large).  Inspect d_treated and d_untreated individually for the full
    #    picture; the selectivity scatter plot visualises both axes.
    treated_dist = {g: d for g, d in ranking}
    untreated_dist = {g: d for g, d in untreated_ranking}
    shared_targets = [g for g in treated_dist if g in untreated_dist]
    selectivity_ranking: list[tuple[str, float, float, float]] = []
    for g in shared_targets:
        d_t = treated_dist[g]
        d_u = untreated_dist[g]
        sel = d_u - d_t
        selectivity_ranking.append((g, d_t, d_u, sel))
    selectivity_ranking.sort(key=lambda x: x[3], reverse=True)  # descending

    top10_sel = selectivity_ranking[:10]
    logger.info(
        "[%s] Top-10 selective candidates (d_untreated - d_treated):\n%s",
        method,
        "\n".join(
            f"  {i+1:3d}. {g:<20s}  sel={s:.4f}  d_treated={dt:.4f}  d_untreated={du:.4f}"
            for i, (g, dt, du, s) in enumerate(top10_sel)
        ),
    )

    selectivity_enrichment = enrichment_auc(
        [g for g, *_ in selectivity_ranking], cfg.positive_controls
    )

    # -- Metric 2: treatment condition CH ratio (control cells only) -----------
    logger.info("[%s] Computing treatment-condition CH ratio ...", method)
    is_ctrl = np.array([g in ctrl_label_set for g in gene_target_labels])
    if is_ctrl.sum() < 2:
        logger.warning(
            "[%s] Fewer than 2 control cells — skipping treatment CH.", method
        )
        treatment_ch = {
            "true": float("nan"),
            "random_mean": float("nan"),
            "random_std": float("nan"),
            "ratio": float("nan"),
        }
    else:
        treatment_ch = calinski_harabasz_ratio(
            emb[is_ctrl],
            [condition_labels[i] for i, c in enumerate(is_ctrl) if c],
            n_random_trials=cfg.n_random_trials,
        )
    logger.info("[%s] Treatment CH ratio: %.3f", method, treatment_ch["ratio"])

    # -- Metric 3: gene-target CH ratio (non-control cells, per condition) -----
    logger.info("[%s] Computing gene-target CH ratio ...", method)
    gene_target_ch: dict[str, dict] = {}
    for cond in [cfg.candidate_condition, "Untreated"]:
        mask = np.array(
            [
                condition_labels[i] == cond
                and gene_target_labels[i] not in ctrl_label_set
                for i in range(len(condition_labels))
            ]
        )
        if (
            mask.sum() < 2
            or len(set(gene_target_labels[i] for i, m in enumerate(mask) if m)) < 2
        ):
            logger.warning(
                "[%s] Not enough cells for gene-target CH (%s).", method, cond
            )
            gene_target_ch[cond] = {
                "true": float("nan"),
                "random_mean": float("nan"),
                "random_std": float("nan"),
                "ratio": float("nan"),
            }
        else:
            gene_target_ch[cond] = calinski_harabasz_ratio(
                emb[mask],
                [gene_target_labels[i] for i, m in enumerate(mask) if m],
                n_random_trials=cfg.n_random_trials,
            )
        logger.info(
            "[%s] Gene-target CH ratio (%s): %.3f",
            method,
            cond,
            gene_target_ch[cond]["ratio"],
        )

    # -- Metric 4a: enrichment AUC for positive controls ----------------------
    logger.info("[%s] Computing enrichment AUC ...", method)
    ranked_targets = [g for g, _ in ranking]
    auc = enrichment_auc(ranked_targets, cfg.positive_controls)
    logger.info("[%s] Enrichment AUC: %.4f", method, auc)

    return {
        "ranking": ranking,
        "untreated_ranking": untreated_ranking,
        "selectivity_ranking": selectivity_ranking,
        "selectivity_enrichment_auc": selectivity_enrichment,
        "treatment_ch": treatment_ch,
        "gene_target_ch": gene_target_ch,
        "enrichment_auc": auc,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_selectivity(
    all_results: dict[str, dict],
    positive_controls: list[str],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """
    For each method produce a scatter plot of d_treated (x) vs d_untreated (y).

    Layout
    ------
    - Gray points: all non-control, non-positive gene targets
    - Blue points: top-10 by selectivity (highest d_untreated - d_treated)
    - Red stars:   positive controls (TNFRSF1A, TRADD, JUNB, JUND, NFKB1, NFKB2)
    - Dashed diagonal y=x: points above this line are selective
    - Selectivity (y - x) increases upward relative to the diagonal
    """
    pos_set = set(positive_controls)

    for method, res in all_results.items():
        sel_ranking = res["selectivity_ranking"]  # list of (gene, d_t, d_u, sel)
        if not sel_ranking:
            continue

        genes = [r[0] for r in sel_ranking]
        d_t_arr = np.array([r[1] for r in sel_ranking])
        d_u_arr = np.array([r[2] for r in sel_ranking])
        sel_arr = np.array([r[3] for r in sel_ranking])

        top10_genes = set(genes[:10])

        fig, ax = plt.subplots(figsize=(7, 6))

        # ---- background: all other genes ------------------------------------
        bg_mask = np.array([g not in top10_genes and g not in pos_set for g in genes])
        ax.scatter(
            d_t_arr[bg_mask],
            d_u_arr[bg_mask],
            s=12,
            alpha=0.35,
            color="#aaaaaa",
            linewidths=0,
            zorder=1,
            label="other targets",
        )

        # ---- top-10 by selectivity ------------------------------------------
        top10_mask = np.array([g in top10_genes and g not in pos_set for g in genes])
        ax.scatter(
            d_t_arr[top10_mask],
            d_u_arr[top10_mask],
            s=40,
            alpha=0.85,
            color="#3b82f6",
            linewidths=0,
            zorder=3,
            label="top-10 selective",
        )
        for i, flag in enumerate(top10_mask):
            if flag:
                ax.annotate(
                    genes[i],
                    (d_t_arr[i], d_u_arr[i]),
                    fontsize=6.5,
                    ha="left",
                    va="bottom",
                    xytext=(3, 2),
                    textcoords="offset points",
                    color="#1d4ed8",
                )

        # ---- positive controls ----------------------------------------------
        pos_mask = np.array([g in pos_set for g in genes])
        ax.scatter(
            d_t_arr[pos_mask],
            d_u_arr[pos_mask],
            s=80,
            marker="*",
            color="#ef4444",
            linewidths=0.4,
            edgecolors="#7f1d1d",
            zorder=4,
            label="positive controls",
        )
        for i, flag in enumerate(pos_mask):
            if flag:
                ax.annotate(
                    genes[i],
                    (d_t_arr[i], d_u_arr[i]),
                    fontsize=7,
                    fontweight="bold",
                    ha="right",
                    va="top",
                    xytext=(-3, -2),
                    textcoords="offset points",
                    color="#991b1b",
                )

        # ---- diagonal y = x -------------------------------------------------
        lim = max(d_t_arr.max(), d_u_arr.max()) * 1.08
        ax.plot(
            [0, lim],
            [0, lim],
            "--",
            color="#6b7280",
            linewidth=0.8,
            label="y = x  (sel = 0)",
            zorder=0,
        )
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

        ax.set_xlabel("d_treated  (Treated centroid → anchor)", fontsize=10)
        ax.set_ylabel("d_untreated  (Untreated centroid → anchor)", fontsize=10)
        ax.set_title(
            f"Selectivity scatter — {method}\n"
            r"$\uparrow$ above diagonal = treatment-specific reversal",
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
        ax.set_aspect("equal")

        fig.tight_layout()
        out_path = os.path.join(output_dir, f"selectivity_scatter_{method}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Saved selectivity scatter: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@draccus.wrap()
def main(cfg: PfizerEvalConfig) -> None:
    if not cfg.train_dir:
        raise ValueError("--train_dir must be set")
    if not cfg.h5ad_path:
        raise ValueError("--h5ad_path must be set")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    )
    logger = logging.getLogger(__name__)

    # ---- output directory ---------------------------------------------------
    output_dir = cfg.output_dir or os.path.join(cfg.train_dir, "eval_pfizer")
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Results will be written to %s", output_dir)

    # ---- load training config -----------------------------------------------
    train_config_path = os.path.join(cfg.train_dir, "config.yaml")
    if not os.path.exists(train_config_path):
        raise FileNotFoundError(f"Training config not found: {train_config_path!r}")
    with open(train_config_path) as f:
        train_cfg: TrainConfig = draccus.load(TrainConfig, f)
    logger.info("Loaded training config from %s", train_config_path)

    # ---- device -------------------------------------------------------------
    device = _resolve_device(cfg.device)
    logger.info("Using device: %s", device)

    # ---- load Pfizer h5ad (backed='r' keeps X on disk — obs loads normally) ---
    logger.info("Loading Pfizer h5ad from %s", cfg.h5ad_path)
    adata = ad.read_h5ad(cfg.h5ad_path, backed="r")
    logger.info("Loaded: %d cells × %d genes  (X backed, not in RAM)", *adata.shape)

    # obs is a plain DataFrame — always loaded, but tiny compared to X
    all_cell_ids: list[str] = list(adata.obs.get("cellID", adata.obs.index))
    all_condition: list[str] = list(adata.obs["condition"])
    all_gene_target: list[str] = list(adata.obs["gene_target"])
    all_group = [f"{c}_{g}" for c, g in zip(all_condition, all_gene_target)]

    # ---- optional stratified subsample (indices only — never subset backed adata) --
    if cfg.max_cells is not None and cfg.max_cells < adata.n_obs:
        logger.info("Subsampling to %d cells (stratified) ...", cfg.max_cells)
        selected_indices: list[int] | None = sorted(
            _stratified_subsample(all_cell_ids, all_group, cfg.max_cells)
        )
        cell_ids = [all_cell_ids[i] for i in selected_indices]
        condition = [all_condition[i] for i in selected_indices]
        gene_target = [all_gene_target[i] for i in selected_indices]
        group = [all_group[i] for i in selected_indices]
        logger.info("Using %d cells after subsample", len(selected_indices))
    else:
        selected_indices = None  # encode all rows sequentially
        cell_ids = all_cell_ids
        condition = all_condition
        gene_target = all_gene_target
        group = all_group

    ctrl_label_set = set(cfg.ctrl_labels)

    # ---- load model ---------------------------------------------------------
    model = make_model(train_cfg.model).to(device)
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
    load_weights_from_checkpoint(model, checkpoint_path, device=device)
    model.eval()
    logger.info(
        "Model loaded (%d parameters)", sum(p.numel() for p in model.parameters())
    )

    # ---- encode with sc_evae ------------------------------------------------
    # Clone the training DatasetConfig and override the file path + filters
    # so the eval inherits every preprocessing choice — including the
    # critical gene_order_path + num_hvg pair, which a model trained on
    # HVGs requires to receive correctly-aligned input. Filters disabled
    # so dataset indices align with raw-adata row indices (preserves
    # selected_indices semantics from upstream stratified subsample).
    from dataclasses import replace as _dc_replace

    eval_dataset_cfg = _dc_replace(
        train_cfg.dataset,
        h5ad_path=cfg.h5ad_path,
        h5ad_paths=[],
        min_genes=0,
        min_cells=0,
        donor_col=None,
        pert_mapping_path=None,
        deg_table_path=None,
        # Optional: re-slice the Pfizer h5ad to the model's gene panel at
        # runtime. Required when the VAE was trained on a pre-sliced panel
        # (e.g. scldm2k 2000 genes, ``gene_order_path=null`` in train cfg)
        # and the Pfizer h5ad is the raw ~38k-gene file. Missing genes are
        # zero-filled by the loader's _build_gene_permutation.
        **(
            {"gene_order_path": cfg.gene_order_path}
            if cfg.gene_order_path is not None
            else {}
        ),
        **({"num_hvg": cfg.num_hvg} if cfg.num_hvg is not None else {}),
        **({"load_lazily": cfg.load_lazily} if cfg.load_lazily is not None else {}),
    )
    logger.info(
        "Encoding with sc_evae (apply_normalize=%s, apply_log1p=%s, use_binning=%s, "
        "gene_order_path=%s, num_hvg=%s) ...",
        eval_dataset_cfg.apply_normalize,
        eval_dataset_cfg.apply_log1p,
        eval_dataset_cfg.use_binning,
        eval_dataset_cfg.gene_order_path,
        eval_dataset_cfg.num_hvg,
    )
    evae_emb = _encode_cells(
        model=model,
        train_dataset_cfg=eval_dataset_cfg,
        selected_indices=selected_indices,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
        use_post_quantized=cfg.use_post_quantized,
        use_pre_quantproj=cfg.use_pre_quantproj,
        logger=logger,
    )

    # ---- collect all methods ------------------------------------------------
    # method_name → (embeddings, group_labels, condition_labels, gene_target_labels)
    method_data: dict[str, tuple[np.ndarray, list[str], list[str], list[str]]] = {
        "sc_evae": (evae_emb, group, condition, gene_target),
    }

    # ---- load baseline h5ads (optional) -------------------------------------
    if cfg.baseline_dir:
        baseline_pattern = os.path.join(cfg.baseline_dir, "*_imru_*_processed.h5ad")
        baseline_paths = sorted(glob.glob(baseline_pattern))
        if not baseline_paths:
            logger.warning(
                "baseline_dir %r provided but no *_imru_*_processed.h5ad files found.",
                cfg.baseline_dir,
            )
        for bp in baseline_paths:
            fname = os.path.basename(bp)
            method_name = fname.split("_imru_")[0]
            logger.info("Loading baseline %r from %s", method_name, bp)
            try:
                b_adata = ad.read_h5ad(bp)
                b_cell_ids = list(b_adata.obs.get("cellID", b_adata.obs.index))
                b_id_to_idx = {cid: i for i, cid in enumerate(b_cell_ids)}

                present_mask = [cid in b_id_to_idx for cid in cell_ids]
                missing = sum(1 for p in present_mask if not p)
                if missing > 0:
                    logger.warning(
                        "Baseline %r missing %d / %d cell IDs — evaluating on %d shared cells.",
                        method_name,
                        missing,
                        len(cell_ids),
                        len(cell_ids) - missing,
                    )

                our_indices = [
                    b_id_to_idx[cid] for cid, p in zip(cell_ids, present_mask) if p
                ]
                b_group = [g for g, p in zip(group, present_mask) if p]
                b_condition = [c for c, p in zip(condition, present_mask) if p]
                b_gene_tgt = [g for g, p in zip(gene_target, present_mask) if p]

                b_sub = b_adata[our_indices]
                b_emb = _densify(b_sub.X)
                del b_adata, b_sub
                method_data[method_name] = (b_emb, b_group, b_condition, b_gene_tgt)
                logger.info(
                    "Loaded baseline %r: embedding shape %s", method_name, b_emb.shape
                )
            except Exception as e:
                logger.error(
                    "Failed to load baseline %r: %s — skipping.", method_name, e
                )

    # ---- load compbio ranking (optional) ------------------------------------
    compbio_ranking: list[str] | None = None
    if cfg.compbio_ranking_path:
        logger.info(
            "Loading compbio reference ranking from %s", cfg.compbio_ranking_path
        )
        with open(cfg.compbio_ranking_path) as f:
            compbio_ranking = [line.strip() for line in f if line.strip()]
        # Filter to targets present in our dataset
        our_treated_targets = set(
            gene_target[i]
            for i in range(len(condition))
            if condition[i] == cfg.candidate_condition
            and gene_target[i] not in ctrl_label_set
        )
        compbio_ranking = [t for t in compbio_ranking if t in our_treated_targets]
        logger.info(
            "compbio_direct: %d targets (after filtering to present treated perts)",
            len(compbio_ranking),
        )

    # ---- run metrics for every method ---------------------------------------
    all_results: dict[str, dict] = {}
    for method, (emb, m_group, m_cond, m_gene_tgt) in method_data.items():
        logger.info("=" * 60)
        logger.info("Evaluating method: %s  (embedding shape: %s)", method, emb.shape)
        all_results[method] = _run_metrics(
            method=method,
            embeddings=emb,
            group_labels=m_group,
            condition_labels=m_cond,
            gene_target_labels=m_gene_tgt,
            ctrl_label_set=ctrl_label_set,
            cfg=cfg,
            logger=logger,
        )

    # ---- ranking agreement vs compbio_direct (Metric 4b) --------------------
    ranking_agreement_results: dict[str, dict] = {}
    if compbio_ranking is not None:
        logger.info("Computing ranking agreement vs compbio_direct ...")
        for method, res in all_results.items():
            ranked_targets = [g for g, _ in res["ranking"]]
            ranking_agreement_results[method] = ranking_agreement(
                ranked_targets, compbio_ranking, overlap_ks=(10, 25)
            )
            logger.info(
                "[%s] vs compbio_direct: spearman=%.3f  overlap@10=%.2f  overlap@25=%.2f",
                method,
                ranking_agreement_results[method]["spearman"],
                ranking_agreement_results[method].get("overlap_at_10", float("nan")),
                ranking_agreement_results[method].get("overlap_at_25", float("nan")),
            )

    # ---- save outputs -------------------------------------------------------

    # ranking_results.csv
    ranking_csv_path = os.path.join(output_dir, "ranking_results.csv")
    with open(ranking_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "rank",
                "gene_target",
                "cosine_distance",
                "cosine_similarity",
                "d_untreated",
                "selectivity",
            ],
        )
        writer.writeheader()
        for method, res in all_results.items():
            # Build lookup from the selectivity table for O(1) join
            sel_lookup: dict[str, tuple[float, float]] = {
                g: (du, s) for g, _dt, du, s in res["selectivity_ranking"]
            }
            for rank, (gene_tgt, dist) in enumerate(res["ranking"], start=1):
                du, sel = sel_lookup.get(gene_tgt, (float("nan"), float("nan")))
                writer.writerow(
                    {
                        "method": method,
                        "rank": rank,
                        "gene_target": gene_tgt,
                        "cosine_distance": dist,
                        "cosine_similarity": 1.0 - dist,
                        "d_untreated": du,
                        "selectivity": sel,
                    }
                )
    logger.info("Saved: %s", ranking_csv_path)

    # clustering_metrics.json
    clustering_path = os.path.join(output_dir, "clustering_metrics.json")
    clustering_out: dict[str, dict] = {}
    for method, res in all_results.items():
        clustering_out[method] = {
            "treatment_condition": res["treatment_ch"],
            "gene_target_treated": res["gene_target_ch"].get(
                cfg.candidate_condition, {}
            ),
            "gene_target_untreated": res["gene_target_ch"].get("Untreated", {}),
        }
    with open(clustering_path, "w") as f:
        json.dump(clustering_out, f, indent=2)
    logger.info("Saved: %s", clustering_path)

    # enrichment_auc.json
    enrichment_path = os.path.join(output_dir, "enrichment_auc.json")
    with open(enrichment_path, "w") as f:
        json.dump(
            {m: res["enrichment_auc"] for m, res in all_results.items()}, f, indent=2
        )
    logger.info("Saved: %s", enrichment_path)

    # ranking_agreement.json
    if ranking_agreement_results:
        agreement_path = os.path.join(output_dir, "ranking_agreement.json")
        with open(agreement_path, "w") as f:
            json.dump(ranking_agreement_results, f, indent=2)
        logger.info("Saved: %s", agreement_path)

    # summary.json — all scalar metrics per method
    summary: dict[str, dict] = {}
    for method, res in all_results.items():
        summary[method] = {
            "enrichment_auc": res["enrichment_auc"],
            "selectivity_enrichment_auc": res["selectivity_enrichment_auc"],
            "treatment_ch_ratio": res["treatment_ch"]["ratio"],
            "gene_target_ch_ratio_treated": res["gene_target_ch"]
            .get(cfg.candidate_condition, {})
            .get("ratio", float("nan")),
            "gene_target_ch_ratio_untreated": res["gene_target_ch"]
            .get("Untreated", {})
            .get("ratio", float("nan")),
            "top10_reversion_targets": [g for g, _ in res["ranking"][:10]],
            "top10_selectivity_targets": [
                g for g, *_ in res["selectivity_ranking"][:10]
            ],
        }
        if method in ranking_agreement_results:
            summary[method].update(ranking_agreement_results[method])
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved: %s", summary_path)

    # ---- plots --------------------------------------------------------------
    _plot_selectivity(all_results, cfg.positive_controls, output_dir, logger)

    # ---- print summary table ------------------------------------------------
    logger.info("")
    logger.info("=" * 82)
    logger.info("SUMMARY")
    logger.info("=" * 82)
    logger.info(
        "%-20s  %-8s  %-8s  %-8s  %-8s  %-8s",
        "method",
        "enr_auc",
        "sel_auc",
        "trt_ch",
        "gt_ch_T",
        "gt_ch_U",
    )
    logger.info("-" * 82)
    for method, s in summary.items():
        logger.info(
            "%-20s  %-8.4f  %-8.4f  %-8.3f  %-8.3f  %-8.3f",
            method,
            s.get("enrichment_auc") or float("nan"),
            s.get("selectivity_enrichment_auc") or float("nan"),
            s.get("treatment_ch_ratio") or float("nan"),
            s.get("gene_target_ch_ratio_treated") or float("nan"),
            s.get("gene_target_ch_ratio_untreated") or float("nan"),
        )
    logger.info("=" * 82)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
