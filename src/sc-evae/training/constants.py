"""
Per-dataset metadata for PerturbationDataset.

Keyed by h5ad *basename* (not full path) so the same file resolves consistently
regardless of where it lives on disk.

Fields
------
cell_type:
    Fallback cell-type label applied to every cell in the file when the file's
    ``.obs`` does NOT have a ``cell_type`` column. When the column is present
    the fallback is unused. Set to ``None`` for files that have the column.
pert_col:
    Name of the ``.obs`` column holding perturbation labels for this file.
    Set to ``None`` for observational datasets — every cell is then treated as
    a control cell carrying ``control_label``.
control_label:
    Label that identifies control cells in ``pert_col``. For observational
    datasets (``pert_col is None``) this is the synthetic label assigned to
    every cell.
obs_filters:
    Optional per-file inclusion filter on .obs columns. Example:
    ``{"condition": []}`` declares ``condition`` as a filterable column on
    this file with NO default filter applied (load all cells). YAML can then
    opt-in to filtering via ``DatasetConfig.obs_filters``. An empty list
    means "no default filter"; a non-empty list means "default to these
    values". Columns named here MUST exist in the file's .obs (strict —
    catches typos at init time).
precomputed_log1p_layer:
    Optional name of a pre-computed log1p layer on the file (e.g. "log1p").
    When set, the loader treats the file as already having
    ``normalize_total + log1p`` baked in: it skips the runtime normalize/log1p
    pass and reads the model input from ``adata.layers[layer]`` when
    ``apply_log1p=True``. ``adata.X`` must hold raw counts; the loader uses
    them when ``apply_log1p=False`` (raw-count NB heads).
    The combinations ``(apply_normalize=True, apply_log1p=False)`` and
    ``(apply_normalize=False, apply_log1p=True)`` are not representable on
    such a file and raise at init.
    The full-panel library size for each cell must be saved in
    ``adata.obs["library_size_full"]`` — runtime ``X.sum(axis=1)`` would only
    cover the post-slice gene panel.
    Pre-computation is intended for files where the canonical pipeline is
    "normalize on the full gene panel, then slice to a smaller HVG subset"
    (e.g. scLDM Parse1M). See ``scripts/utils/preprocess_parse_donor1_scldm2k.py``.
    Mutually exclusive with ``derive_log1p_via_lib_full``.
derive_log1p_via_lib_full:
    Boolean. When True, the file holds raw counts on a pre-sliced HVG panel,
    and log1p is *derived* at read time from
    ``log1p(uns['preprocessing']['median_library_size_full']
            / obs['library_size_full'][cell] * X[cell, gene])``.
    This is mathematically identical to caching a log1p layer (the same
    formula was used at preprocess time) but saves ~half the file size,
    speeds up lazy reads (less I/O per cell), and lets us gzip-compress the
    raw matrix without paying decompression cost on a redundant float layer.
    Requires ``adata.obs['library_size_full']`` and
    ``adata.uns['preprocessing']['median_library_size_full']`` to be present.
    The library size returned for downstream NB / CP10K paths is
    ``library_size_full`` (full-panel) — NOT the post-slice row sum.
    Same ``apply_log1p`` / ``apply_normalize`` mutual constraints as
    ``precomputed_log1p_layer``. Mutually exclusive with that field.
    (Not used by the released Replogle/Parse files, which ship a cached
    ``log1p`` layer via ``precomputed_log1p_layer``; kept as a generic loader
    capability.)

Files not listed here fall back to ``DatasetConfig.pert_col`` /
``DatasetConfig.ctrl_label`` and require a ``cell_type`` column in ``.obs``
(otherwise an error is raised at init time).
"""

from typing import Any

DATASET_METADATA: dict[str, dict[str, Any]] = {
    "replogle_combined_scldm2k.h5ad": {
        # scLDM-faithful Replogle-Nadig combined: all 4 cell lines (hepg2,
        # k562, jurkat, rpe1) merged into one file × scLDM's 2,000-gene HVG
        # panel from metadata/replogle_test.json. normalize_total(median) +
        # log1p was applied per-file on the full ~9k-gene panel offline,
        # then sliced to the 2k scLDM panel (in scLDM JSON order).
        # Produced by scripts/utils/preprocess_replogle_scldm2k.py.
        # adata.X = raw counts (int32, sparse); adata.layers["log1p"] =
        # full-panel-normalized log1p (float32, sparse). adata.obs has
        # 'cell_type' (cell line, lowercase: hepg2/k562/jurkat/rpe1),
        # 'gene' (CRISPR target perturbation, HGNC symbol), and
        # 'library_size_full' (per-cell ~9k-panel row sum). var.index =
        # HGNC symbol; var["gene_id"] = canonical Ensembl ID.
        "cell_type": None,
        "pert_col": "gene",
        "control_label": "non-targeting",
        "precomputed_log1p_layer": "log1p",
    },
    "parse1m_donor1_scldm2k.h5ad": {
        # scLDM-faithful Parse-PBMC subset: Donor1 only (1,267,690 cells)
        # × scLDM's published 2,000-gene HVG panel. normalize_total(median)
        # + log1p was applied on the FULL 40k panel offline, then sliced
        # to 2k. Produced by scripts/utils/preprocess_parse_donor1_scldm2k.py.
        # adata.X = raw counts (int32, sparse) on the 2k slice; adata.layers
        # ["log1p"] = log1p of full-panel-normalized counts on the 2k slice.
        # adata.obs has 'cell_type', 'cytokine', 'donor' (single value
        # "Donor1"), and 'library_size_full' (full 40k-panel row sums).
        "cell_type": None,
        "pert_col": "cytokine",
        "control_label": "PBS",
        "precomputed_log1p_layer": "log1p",
    },
    "pfizer_raw.h5ad": {
        "cell_type": "T_cell",
        "pert_col": "gene_target",
        "control_label": "SAFE_TARGET",
        # Default: load BOTH treated and untreated. YAML can opt in to
        # ``obs_filters: {condition: [Untreated]}`` to restrict.
        "obs_filters": {"condition": []},
    },
    # Replogle-Nadig (Nadig et al. 2025, GEO GSE264667) cell-line files —
    # downloaded via scripts/utils/download_dataset.py into datasets/replogle/.
    # Same .obs schema as Replogle 2022 (gene + non-targeting control).
    "K562_essential_raw_singlecell_01.h5ad": {
        "cell_type": "K562",
        "pert_col": "gene",
        "control_label": "non-targeting",
    },
    "hepg2_raw_singlecell_01.h5ad": {
        "cell_type": "hepg2",
        "pert_col": "gene",
        "control_label": "non-targeting",
    },
    "jurkat_raw_singlecell_01.h5ad": {
        "cell_type": "jurkat",
        "pert_col": "gene",
        "control_label": "non-targeting",
    },
    "rpe1_raw_singlecell_01.h5ad": {
        "cell_type": "rpe1",
        "pert_col": "gene",
        "control_label": "non-targeting",
    },
}

# ---------------------------------------------------------------------------
# Precomputed DEG-table cache paths
# ---------------------------------------------------------------------------
#
# DEG tables are deterministic given (cells, ctrl_label, top_k) and the
# numeric values are config-invariant up to small quantization (because
# train_utils.build_deg_table_from_dataset routes through to_metric_space).
# Recomputing the same artifact for every training run wastes minutes-to-hours
# on Parse and reploall_combined. These maps point ``DatasetConfig`` at a
# precomputed .pt file produced by ``scripts/utils/precompute_deg_table.py``.
#
# Resolution order in train.py:
#   1. DatasetConfig.deg_table_path (YAML override)
#   2. resolve_default_deg_table_path(dataset_cfg) — these dicts
#   3. recompute on the fly (no path resolved)
#
# Loader applies a shape check at load time:
#   * mask width  == ds.num_genes
#   * indices len == top_k
# Mismatch → load fails → caller can recompute or error.

DEFAULT_DEG_TABLE_PATHS: dict[str, str] = {
    # single-file: keyed by h5ad basename
    "parse1m_donor1_scldm2k.h5ad": "assets/deg_tables/parse1m_donor1_scldm2k_top20.pt",
    "replogle_combined_scldm2k.h5ad": "assets/deg_tables/replogle_combined_scldm2k_top20.pt",
}

DEFAULT_DEG_TABLE_PATHS_MULTIFILE: dict[frozenset[str], str] = {
    # multi-file: keyed by the frozenset of file basenames in the config.
    # Four-cell-line Replogle-Nadig combined dataset (raw per-cell-line files,
    # used when reproducing the scLDM-2k preprocessing from raw):
    frozenset(
        {
            "K562_essential_raw_singlecell_01.h5ad",
            "hepg2_raw_singlecell_01.h5ad",
            "jurkat_raw_singlecell_01.h5ad",
            "rpe1_raw_singlecell_01.h5ad",
        }
    ): "assets/deg_tables/replogle_combined_top20.pt",
}


def resolve_default_deg_table_path(
    h5ad_path: str | None, h5ad_paths: list[str] | None
) -> str | None:
    """Look up a default DEG-table cache path for the given h5ad input.

    Returns the registered path or None. Falls through to the multi-file
    map when ``h5ad_paths`` is non-empty, otherwise the single-file map.
    """
    import os

    if h5ad_paths:
        key = frozenset(os.path.basename(p) for p in h5ad_paths)
        return DEFAULT_DEG_TABLE_PATHS_MULTIFILE.get(key)
    if h5ad_path:
        return DEFAULT_DEG_TABLE_PATHS.get(os.path.basename(h5ad_path))
    return None


# ---------------------------------------------------------------------------
# Precomputed real-DE-table cache paths (Mann-Whitney U + BH-FDR, long-form)
# ---------------------------------------------------------------------------
#
# The cell-eval metrics ingest a per-(pert, gene) DE table built by
# ``sc_evae.metrics.de_table.build_de_table``. Computing it requires running a
# vectorised Mann-Whitney U test across thousands of cells × thousands of
# genes per pert and is the dominant single-CPU cost in eval.py for large
# datasets. The output depends only on (val cells, ctrl cells, gene_names,
# de_epsilon) — not on the model — so when many checkpoints share the same
# val split it can be cached.
#
# Resolution order in eval.py:
#   1. EvalConfig.de_table_path (YAML / CLI override)
#   2. resolve_default_de_table_path(dataset_cfg, split_stem, de_epsilon)
#      — these dicts
#   3. compute on the fly + write to the resolved cache path (when one
#      resolves) so subsequent runs hit it.
#
# Loader applies a coverage check at load time:
#   * cached gene set ⊇ current gene_names
#   * cached target set ⊇ val perts to be scored
# Mismatch → falls back to recompute (does not error).

DEFAULT_DE_TABLE_PATHS: dict[tuple[str, str | None, float], str] = {
    # Keyed by (h5ad basename, split-spec JSON stem, de_epsilon). The split
    # stem is the basename of DatasetConfig.split_path without its .json suffix
    # (e.g. assets/scldm_replogle_test.json → "scldm_replogle_test"). Add new
    # rows when introducing a fixed split + epsilon combination that benefits
    # from cross-checkpoint caching.
    # Principled per-dataset epsilon: replogle (HEPG2 holdout) has dense
    # controls — no zero-ctrl genes — so eps=0.0 is used to match cell-eval /
    # STATE faithfully. parse1m (CD4-Naive holdout) has sparse controls
    # (zero-ctrl genes → ±inf log2FC at eps=0), so eps=0.5 is used to keep the
    # DE-LFC metrics well-defined. Run eval with --de_epsilon matching the
    # dataset (0.0 for replogle, 0.5 for parse1m) to hit these caches.
    (
        "replogle_combined_scldm2k.h5ad",
        "scldm_replogle_test",
        0.0,
    ): "assets/de_tables/replogle_combined_scldm2k_hepg2_eps0.0.csv.gz",
    (
        "replogle_combined_scldm2k.h5ad",
        "scldm_replogle_test",
        0.5,
    ): "assets/de_tables/replogle_combined_scldm2k_hepg2_eps0.5.csv.gz",
    (
        "parse1m_donor1_scldm2k.h5ad",
        "scldm_parse1m_test",
        0.5,
    ): "assets/de_tables/parse1m_donor1_scldm2k_cd4_naive_eps0.5.csv.gz",
    # parse1m eps=0.0 also available (cell-eval-faithful); LFC has many ±inf
    # from sparse CD4-Naive controls, which the hardened DE-LFC metrics omit.
    (
        "parse1m_donor1_scldm2k.h5ad",
        "scldm_parse1m_test",
        0.0,
    ): "assets/de_tables/parse1m_donor1_scldm2k_cd4_naive_eps0.0.csv.gz",
}


def split_stem(split_path: str | None) -> str | None:
    """Cache-key stem for a split spec: basename without the ``.json`` suffix.

    ``assets/scldm_replogle_test.json`` → ``"scldm_replogle_test"``.
    ``None`` (random split) → ``None``.
    """
    import os

    if split_path is None:
        return None
    return os.path.splitext(os.path.basename(split_path))[0]


def resolve_default_de_table_path(
    h5ad_path: str | None,
    h5ad_paths: list[str] | None,
    split_path: str | None,
    de_epsilon: float,
) -> str | None:
    """Look up a default real-DE-table cache path.

    ``split_path`` is the ``DatasetConfig.split_path`` value; it is reduced to
    its JSON stem (see ``split_stem``) to form the cache key. Returns the
    registered path or None. Multi-file inputs are not cached by default (no
    canonical paper split is published over multi-file Replogle/Parse layouts
    at present); for those, set ``EvalConfig.de_table_path`` explicitly.
    """
    import os

    if h5ad_paths and not h5ad_path:
        return None
    if h5ad_path is None:
        return None
    return DEFAULT_DE_TABLE_PATHS.get(
        (os.path.basename(h5ad_path), split_stem(split_path), float(de_epsilon))
    )
