"""
EvalConfig — configuration for scripts/eval.py.

Minimal eval config: point at a training directory and the script reads the
full TrainConfig (dataset path, model architecture, etc.) from
``{train_dir}/config.yaml`` automatically.  Only eval-specific knobs live here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalConfig:
    # ---- required -----------------------------------------------------------
    train_dir: str = ""
    """Path to the experiment directory produced by scripts/train.py.
    The script reads ``{train_dir}/config.yaml`` to recover the dataset path,
    model architecture, split seed, and all other training settings."""

    # ---- checkpoint selection -----------------------------------------------
    checkpoint: str | None = None
    """Name of the checkpoint sub-directory inside ``{train_dir}/checkpoints/``
    (e.g. ``step_0010000``).  When *None* the latest checkpoint is used."""

    # ---- output -------------------------------------------------------------
    output_dir: str | None = None
    """Directory where results are written.  Defaults to
    ``{train_dir}/eval_results/`` when *None*."""

    # ---- inference settings -------------------------------------------------
    batch_size: int = 512
    num_workers: int = 4
    device: str = "auto"
    """``"auto"`` selects CUDA if available, otherwise CPU.
    Can also be ``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc."""

    # ---- dataset overrides --------------------------------------------------
    ctrl_label: str | None = None
    """Override the ctrl_label from the training config (e.g. 'non-targeting').
    When None, the value from {train_dir}/config.yaml is used."""

    split_path: str | None = None
    """Override DatasetConfig.split_path from the training config. Lets you
    evaluate a checkpoint trained with a random val split on a cell-type × pert
    holdout (e.g. 'assets/scldm_replogle_test.json') without rewriting the saved
    config.yaml. When None, the saved value is used."""

    # ---- metric knobs -------------------------------------------------------
    top_k_deg: int = 20
    """Top-k DEGs per pert in the DEG table (shape-checked against any cached
    table; the table provides ctrl_mean and the top-k mask used by the
    train-time proxy metrics)."""

    max_cells_per_pert: int = 1000
    """Cap on cells kept in memory per perturbation during the inference loop.
    Each accumulated cell contributes a ``(num_genes,)`` prediction + ground-
    truth tensor; without a cap, held-out splits with millions of cells exhaust
    host RAM. 1000 is well above the sample-size floor of every metric
    (per-pert means saturate at ~100-200 cells; DE Mann-Whitney p-values are
    stable from ~500 cells/pert; distribution metrics already cap at
    ``distribution_max_samples=5000`` per group inside their kernels) and is
    ~4x what cell-eval / State benchmarks use by default. Bump to 2000-5000
    if a specific metric looks unstable. Set to 0 to disable the cap.

    Does NOT apply to control cells — see ``max_ctrl_cells``."""

    max_ctrl_cells: int = 5000
    """Cap on control (reference) cells used as the DE-table denominator and
    pseudobulk reference. Decoupled from ``max_cells_per_pert`` because:

    - The ctrl pool is shared across all perts, so making it large is cheap.
    - Per-gene LFC and Mann-Whitney p-values are dominated by ctrl-side
      sparsity: with HVG slicing, many genes have zero raw count in any
      given cell, so 1000 cells is too few to estimate a reliable per-gene
      mean (43% of genes can come out all-zero on rare-cell-type ctrl pools).
      All-zero ctrl-pseudobulks → log2fc = ±inf → de_spearman_lfc_sig and
      mae_log2fc_sig collapse to noise.
    - State / scLDM eval pipelines effectively use thousands of ctrl cells.

    Applied both to ctrl cells from the val split (subsample if val has more
    than this) and to the train-fallback ctrl pool."""

    de_epsilon: float = 0.0
    """Pseudocount added to both ``mean_pert`` and ``mean_ctrl`` before the
    log2 fold change is computed in ``build_de_table``:
    ``log2_fold_change = log2((mean_pert + ε) / (mean_ctrl + ε))``.

    Default 0.0 matches **STATE / cell-eval / pdex exactly**: STATE runs
    cell-eval with ``pdex_kwargs=dict(exp_post_agg=True, is_log1p=True)`` and
    never overrides ``epsilon``, so pdex's default ``epsilon=0.0`` applies
    (state/.../_predict.py:584; pdex/_math.py:109). Use 0.0 to directly
    compare against their reported DE-LFC numbers.

    pdex's *docstring* recommends 0.5 for sparse CRISPRi/CRISPRa screens, and
    0.5 does dampen the ±inf LFCs that arise on rare-cell-type controls (43%
    of HVGs can have zero ctrl pseudobulk on CD4-Naive PBS cells), which
    otherwise make ``de_spearman_lfc_sig`` / ``mae_log2fc_sig`` collapse to
    noise on the Parse cell-type holdout. STATE itself used 0.0, so 0.0 is the
    faithful default; pass ``--de_epsilon 0.5`` on the Parse holdout if those
    two LFC metrics look unstable (this deviates from STATE but stabilizes
    sparse-control LFCs)."""

    de_table_path: str | None = None
    """Path to a precomputed real-DE table (gzipped CSV produced by a prior
    eval run, ``real_de_table.csv.gz`` shape). When the file exists, the eval
    loop loads + validates it (gene set + pert coverage + epsilon match) and
    skips the expensive Mann-Whitney U + BH-FDR rebuild — only ``pred_de`` is
    rebuilt per checkpoint. Useful for parallel RunPod evals where many model
    checkpoints share the same val split.

    When None, falls back to ``constants.resolve_default_de_table_path(
    h5ad_path, h5ad_paths, split_path, de_epsilon)``. If neither resolves to
    an existing file, ``real_de`` is built fresh and (when a cache path is
    resolvable) written atomically to seed the cache for subsequent runs."""

    n_ch_random_trials: int = 30
    """Number of shuffled-label trials used to compute the Calinski-Harabasz
    ratio baseline.  Higher = more stable estimate; 30 matches the Pfizer eval."""

    num_sample_steps: int | None = 32
    """Override the sampler step count for MDLM / Flow.  Defaults to 32 — for
    a 32-token latent, MDLM ancestral unmasking saturates well before then,
    and Heun on a straight-line CFM in (32, 8) latent is within noise of
    higher step counts. Pass an explicit value (or set to ``None`` from a
    YAML override) to fall back to the training-time value."""

    # ---- scLDM distribution-level metrics ----------------------------------
    pca_components: int = 30
    """PCA dimensionality for distribution-level metrics (W2, MMD² RBF, FD).
    Matches scLDM Table 3 (Appendix J.5: ``calculated to 30 principal
    components``). PCA is fit on the pooled TRUE cells; generated cells are
    projected through the same loadings."""

    distribution_max_samples: int = 5_000
    """Maximum cells per group used in W2 / MMD² RBF / FD. Exact OT (W2 EMD)
    scales as O(n³); subsample beyond this to keep runtime bounded."""

    w2_method: str = "emd"
    """``"emd"`` (default) — exact 2-Wasserstein via ``ot.emd2``.

    NOTE: ``"sinkhorn"`` is intentionally NOT the default. scLDM's released
    code uses ``ot.sinkhorn2(reg=0.05)`` on the *full-gene* cost matrix, but on
    our PCA-30 squared-distance costs ``reg=0.05`` is far smaller than the cost
    scale, so the Sinkhorn kernel ``exp(-M/reg)`` underflows to 0 (POT raises
    divide-by-zero / "numerical errors at iteration 0") and W2 collapses to
    ~0 — a spuriously "perfect" score. EMD is exact and stable here, so it is
    the default. Only switch to ``"sinkhorn"`` with a cost-appropriate ``reg``
    (and log-domain stabilization), not the raw 0.05."""

    seeds: int = 1
    """Reserved for multi-seed aggregation (scLDM reports mean ± std across
    3 sampling seeds). Currently a no-op — eval.py runs one inference pass.
    To get scLDM-style 3-seed std, run eval.py three times with different
    `--seed_offset` values (set via env var or your launcher) and post-
    process the three `summary_metrics.json` files."""

    # ---- metric-space basis -------------------------------------------------
    wandb_upload: bool = False
    """When True, attach the eval_results dir as a W&B artifact to the
    *existing* training run (matched by ``train_cfg.run_name`` in
    ``train_cfg.wandb.entity/project``). The eval is uploaded via
    ``wandb.init(id=<existing_run_id>, resume="must")`` + ``log_artifact``
    + ``finish()`` — no new run is created and no scalars are logged, so
    the artifact appears under the training run's "Artifacts" tab without
    cluttering the project's run list. Falls back silently when the
    training run isn't found (logs a warning).

    Used by the RunPod eval bootstrap to ship results back without a
    pull-from-pod step. Default False so local eval runs don't touch W&B.
    Set to True via CLI / env / YAML override on the eval pod."""

    cp10k_basis: str = "hvg_row_sum"
    """Which per-cell library to normalize against in ``to_metric_space``.

    - ``"hvg_row_sum"`` (default) — divides by the HVG row sum of each cell so
      its HVG genes sum to 1e4 (HVG-fractional CP10K). Matches scLDM / State
      paper convention: their data pipeline ships pre-HVG-sliced files, so
      their ``library_size`` *is* the HVG row sum. Use this to reproduce
      paper-comparable numbers.
    - ``"full_library"`` — divides by the full pre-HVG transcriptome library
      (captured by the data loader before HVG slicing). scVI / scanpy
      convention; useful for cross-dataset comparisons or directly
      interpretable per-gene values. NOTE: the absolute scale is the loader's
      normalization target (``dataset.target_sum``, or the per-file median
      when ``None``) — so it is CP10K only when ``target_sum=1e4``, otherwise
      counts-per-target; use it only for scale-invariant metrics.

    log1p- / apply_normalize-trained models have full-pre-HVG-library
    normalization baked in at preprocessing (counts-per-target, where target is
    the per-file median by default — NOT necessarily 1e4). The ``hvg_row_sum``
    basis applies a per-cell rescale (``full_lib / HVG_lib``, computed from the
    cell's own HVG row sum) before log1p, which cancels both that target and
    the full-vs-HVG library factor and brings those models onto the same metric
    space as raw-counts-trained models. Under ``hvg_row_sum`` every output head
    collapses to the identical scale (``log1p(raw_g / HVG_row_sum * 1e4)``), so
    cross-model comparison is exactly 1:1; under ``full_library`` the absolute
    scale differs by config, so only scale-invariant metrics are comparable."""
