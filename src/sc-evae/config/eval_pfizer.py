"""
PfizerEvalConfig — configuration for scripts/eval_pfizer.py.

Points at a training directory (reads model arch + preprocessing from config.yaml)
and a Pfizer IMRU h5ad file to evaluate OOD phenotypic reversion and embedding
quality metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PfizerEvalConfig:
    # ---- required -----------------------------------------------------------
    train_dir: str = ""
    """Path to the experiment directory produced by scripts/train.py.
    Reads ``{train_dir}/config.yaml`` to recover model architecture and
    preprocessing settings."""

    h5ad_path: str = ""
    """Path to the Pfizer h5ad — the raw ~38k-gene file
    (``datasets/pfizer/pfizer_raw.h5ad``), sliced to the model's gene panel on
    the fly via ``gene_order_path`` + ``num_hvg`` (no separate alignment step).
    A file pre-aligned to the model's panel also works (pass it directly,
    without ``gene_order_path``). Must have .obs columns: condition,
    gene_target, cellID."""

    gene_order_path: str | None = None
    """Override ``train_cfg.dataset.gene_order_path`` to slice the Pfizer h5ad
    to a specific gene panel at runtime. Use when the trained VAE expects a
    pre-sliced panel (e.g. scldm2k 2000-gene panel) but the Pfizer file is
    unsliced raw counts. Genes in the panel that are missing from the Pfizer
    h5ad are zero-filled (the loader's ``_build_gene_permutation`` handles
    this). Set to one of:
      - ``assets/replogle_scldm2k_gene_order.txt`` (2000 genes,
        scldm-faithful Replogle panel — for VAEs trained on
        ``replogle_combined_scldm2k.h5ad``)
      - ``assets/parse1m_donor1_scldm2k_gene_order.txt`` (2000 genes,
        scldm-faithful Parse panel — for VAEs trained on
        ``parse1m_donor1_scldm2k.h5ad``)
    Leave None to inherit from the training config (correct for legacy
    VAEs trained on Pfizer-pre-aligned files)."""

    num_hvg: int | None = None
    """Override ``train_cfg.dataset.num_hvg``. Typically set to the same
    length as ``gene_order_path`` (e.g. 2000 for scldm2k) — the loader
    permutes to the gene order and then slices to top ``num_hvg``. Leave
    None to inherit from the training config."""

    load_lazily: bool | None = None
    """Override ``train_cfg.dataset.load_lazily``. Typically set to True
    when the Pfizer h5ad is large (``pfizer_raw.h5ad`` is ~53 GB on disk
    and would densify to ~133 GB — OOM on a 60 GB host). Lazy mode reads
    each row on-demand from the backed h5ad file (fork-safe across
    DataLoader workers) and supports ``gene_order_path`` permutation at
    row-read time. Leave None to inherit from the training config."""

    # ---- checkpoint selection -----------------------------------------------
    checkpoint: str | None = None
    """Name of the checkpoint sub-directory inside ``{train_dir}/checkpoints/``
    (e.g. ``step_0010000``). When None the latest checkpoint is used."""

    # ---- optional baselines -------------------------------------------------
    baseline_dir: str | None = None
    """Directory containing ``{method}_imru_full_processed.h5ad`` files from the
    Pfizer data release. When provided, each file's ``adata.X`` is used as the
    baseline embedding for that method. If not provided, only sc_evae is evaluated."""

    compbio_ranking_path: str | None = None
    """Path to a ranked target list (.txt, one gene per line) from the Pfizer
    zenodo deposit (e.g. ``20251203_perturbseq_ranked_list_direct_method_trad.txt``).
    When provided, ranking agreement metrics (Spearman, overlap @10/25) are
    computed between sc_evae's ranking and this expert reference."""

    # ---- output -------------------------------------------------------------
    output_dir: str | None = None
    """Directory where results are written. Defaults to
    ``{train_dir}/eval_pfizer/`` when None."""

    # ---- inference settings -------------------------------------------------
    device: str = "auto"
    """``"auto"`` selects CUDA if available, otherwise CPU."""

    batch_size: int = 512
    num_workers: int = 4

    # ---- sampling -----------------------------------------------------------
    max_cells: int | None = None
    """Stratified subsample cap (proportional per {condition}_{gene_target} group).
    None = use all cells, matching the Pfizer paper. Set to a small value
    (e.g. 5000) for quick smoke tests."""

    # ---- VAE embedding choice -----------------------------------------------
    use_post_quantized: bool = False
    """When False (default): use the pre-quantization continuous latent from
    ``model.encode(counts)`` — richer signal, better for cosine distance on OOD data.
    When True: use post-quantization codes ``model.quantizer(...).codes`` instead."""

    use_pre_quantproj: bool = False
    """When True: mean-pool the transformer encoder output *before* the
    ``quantizer_proj_in`` MLP across latent tokens, yielding a single
    ``d_model``-dimensional vector (e.g. 256).  This is a much richer
    representation than the post-projection bottleneck (e.g. 32×3 = 96 for
    FSQ) and may yield better distance-based metrics.
    Ignored when ``use_post_quantized`` is True."""

    # ---- metric knobs -------------------------------------------------------
    n_random_trials: int = 30
    """Number of shuffled-label trials used to compute the Calinski-Harabasz ratio
    denominator (true score / mean random score)."""

    ctrl_labels: list[str] = field(default_factory=lambda: ["SAFE_TARGET", "NO-TARGET"])
    """Gene target labels that identify control (non-perturbed) cells."""

    positive_controls: list[str] = field(
        default_factory=lambda: ["TNFRSF1A", "TRADD", "JUNB", "JUND", "NFKB1", "NFKB2"]
    )
    """Known anti-inflammatory NFkB pathway genes used as positive controls for
    enrichment AUC. These are the 'internal expertise derived (+) control anchors'
    from the Pfizer paper (analysis.py). NOT the same as compbio_direct."""

    anchor_state: str = "Untreated_SAFE_TARGET"
    """The healthy reference centroid label (format: {condition}_{gene_target})."""

    candidate_condition: str = "Treated"
    """Which treatment condition's perturbations to rank for phenotypic reversion."""
