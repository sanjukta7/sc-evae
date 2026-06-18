"""
perturbation_loader.py
----------------------
DataLoader for single-cell perturbation data (.h5ad).

Supports a single file (``config.h5ad_path``) or a list of files
(``config.h5ad_paths``) loaded together; multi-file mode concatenates cells
across files and mixes them into batches. All files must share the same gene
set (same ``n_vars`` and ``var_names`` order).

Per-file metadata (cell type fallback, pert column, control label) is looked
up by basename in ``sc_evae.training.constants.DATASET_METADATA``. If the file's
``.obs`` has a ``cell_type`` column it is used directly; otherwise the fallback
from the metadata table is applied to every cell. Files with ``pert_col=None``
in the metadata table are treated as observational (every cell is a control
cell carrying ``control_label``).

Two loading modes:
  - Eager (default, load_lazily=False): loads the entire count matrix into RAM
    as a dense float32 numpy array. Fast random access; not suitable for very
    large datasets (>~20 GB).
  - Lazy (load_lazily=True): never loads X into RAM. Each DataLoader worker
    opens its own HDF5 file handle per source file on the first __getitem__
    call (fork-safe). Preprocessing (normalize_total, log1p) is applied per-row.

Output keys
~~~~~~~~~~~
  counts        (batch, n_genes) float32 — preprocessed gene expression
  pert_idx      (batch,)         int64   — index into unique_perts
  cell_type     list[str]                — per-cell cell-type label
  cell_type_idx (batch,)         int64   — index into unique_cell_types
  pert_emb      (batch, emb_dim) float32 — only when pert_mapping_path is given

Usage
~~~~~
  # Single file
  ds = PerturbationDataset(config=DatasetConfig(h5ad_path="cells.h5ad"))

  # Multiple files mixed into the same batches
  ds = PerturbationDataset(
      config=DatasetConfig(h5ad_paths=["cells_a.h5ad", "cells_b.h5ad"]),
  )

  # Train / val split (all heavy arrays shared by reference)
  train_ds, val_ds = split_dataset(ds, val_frac=0.1, seed=42)
  train_dl = make_dataloader_from_dataset(train_ds, batch_size=512, num_workers=8)
  val_dl   = make_dataloader_from_dataset(val_ds,   batch_size=512, num_workers=4, shuffle=False)
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# anndata warns about non-unique var_names when opening certain h5ad files
# (e.g. Pfizer dataset). The names are intentionally duplicated upstream;
# calling var_names_make_unique would silently rename genes. Suppress here.
warnings.filterwarnings(
    "ignore",
    message="Variable names are not unique",
    category=UserWarning,
    module="anndata",
)

import numpy as np
import torch
from sc_evae.config.dataset import DatasetConfig
from sc_evae.training.constants import DATASET_METADATA
from sc_evae.training.tokenizer import (
    COUNT_BIN_NUM_CLASSES,
    COUNT_BIN_UPPER,
    _BINNING_MODES,
    _apply_binning,
    _binning_count_max,
)
from torch.utils.data import DataLoader, Dataset


def _resolve_h5ad_paths(config: DatasetConfig) -> list[str]:
    """Resolve ``config`` into a non-empty list of h5ad paths."""
    if config.h5ad_paths:
        if config.h5ad_path:
            raise ValueError(
                "Set either DatasetConfig.h5ad_path (single) or "
                "DatasetConfig.h5ad_paths (list), not both."
            )
        return list(config.h5ad_paths)
    if config.h5ad_path:
        return [config.h5ad_path]
    raise ValueError(
        "DatasetConfig requires either h5ad_path or a non-empty h5ad_paths list."
    )


def _resolve_file_metadata(h5ad_path: str, config: DatasetConfig) -> dict[str, Any]:
    """
    Resolve per-file metadata. Returns ``pert_col``, ``control_label``, and
    ``cell_type_fallback`` (the single label applied when the file has no
    ``cell_type`` obs column). Files not listed in ``DATASET_METADATA`` inherit
    ``pert_col`` and ``control_label`` from the global DatasetConfig.
    """
    basename = os.path.basename(h5ad_path)
    if basename in DATASET_METADATA:
        entry = DATASET_METADATA[basename]
        pert_col = entry["pert_col"]
        control_label = entry["control_label"]
        cell_type_fallback = entry["cell_type"]
        per_file_obs_filters: dict[str, list[str]] = dict(
            entry.get("obs_filters") or {}
        )
        precomputed_log1p_layer: str | None = entry.get("precomputed_log1p_layer")
        derive_log1p_via_lib_full: bool = bool(
            entry.get("derive_log1p_via_lib_full", False)
        )
    else:
        pert_col = config.pert_col
        control_label = config.ctrl_label
        cell_type_fallback = None
        per_file_obs_filters = {}
        precomputed_log1p_layer = None
        derive_log1p_via_lib_full = False
    if precomputed_log1p_layer is not None and derive_log1p_via_lib_full:
        raise ValueError(
            f"DATASET_METADATA[{basename!r}] sets both precomputed_log1p_layer "
            f"and derive_log1p_via_lib_full — they are mutually exclusive."
        )

    # Resolve the effective obs_filters for this file: per-file defaults,
    # column-level-overridden by config.obs_filters.
    resolved: dict[str, list[str]] = dict(per_file_obs_filters)
    strict_cols = set(per_file_obs_filters.keys())
    for col, vals in (config.obs_filters or {}).items():
        resolved[col] = list(vals)

    return {
        "basename": basename,
        "pert_col": pert_col,
        "control_label": control_label,
        "cell_type_fallback": cell_type_fallback,
        "obs_filters": resolved,
        "obs_filter_strict_cols": strict_cols,
        "precomputed_log1p_layer": precomputed_log1p_layer,
        "derive_log1p_via_lib_full": derive_log1p_via_lib_full,
    }


def _build_obs_filter_mask(
    obs,
    resolved_filters: dict[str, list[str]],
    strict_cols: set[str],
    basename: str,
    logger,
) -> np.ndarray:
    """Return a boolean mask of length ``len(obs)`` from the resolved filter dict.

    - Empty list for a column → no-op for that column.
    - Column missing on the file → ValueError if column is in ``strict_cols``
      (declared per-file), else warn and skip.
    """
    mask = np.ones(len(obs), dtype=bool)
    for col, vals in resolved_filters.items():
        if not vals:
            continue
        if col not in obs.columns:
            if col in strict_cols:
                raise ValueError(
                    f"obs_filters: file {basename!r} declared column {col!r} in "
                    f"DATASET_METADATA but the file's .obs lacks it."
                )
            try:
                logger.warning(
                    f"  [{basename}] obs_filters: column {col!r} not in file's "
                    f"obs; skipping filter for this file.",
                    main_process_only=True,
                )
            except TypeError:
                logger.warning(
                    f"  [{basename}] obs_filters: column {col!r} not in file's "
                    f"obs; skipping filter for this file."
                )
            continue
        col_vals = obs[col].astype(str).to_numpy()
        accept = set(str(v) for v in vals)
        col_mask = np.fromiter(
            (v in accept for v in col_vals), dtype=bool, count=len(col_vals)
        )
        mask &= col_mask
    return mask


@dataclass
class DataStats:
    """
    Summary statistics for a PerturbationDataset (or a train/val split).

    Depth / library-size fields are in **raw count units**, computed from each
    cell's raw library size (before normalize_total / log1p) — not from the
    preprocessed model-input values.

    Contract: only ``log_lib_mean`` / ``log_lib_std`` are consumed by code (the
    log-normal library prior for generation, read back from dataset_stats.json
    by ``factory.load_vae_from_run``). The rest are diagnostics written to
    dataset_stats.json for inspection.
    """

    # --- dataset shape ---
    num_cells: int
    num_genes: int
    num_perts: int

    # --- log-normal prior parameters for l_n (over raw library size) ---
    log_lib_mean: float  # mean of log(raw library size)
    log_lib_std: float  # std  of log(raw library size)

    # --- normalization reference ---
    # Global median of raw library sizes. With ``DatasetConfig.target_sum=None``
    # this is the scale cells are normalized toward (per-file at load time; this
    # is the dataset-wide median for reference).
    median_library_size: float

    # --- diagnostics ---
    count_max: float  # resolved max integer count (sizes count embeddings)
    depth_mean: float  # mean raw library size
    depth_median: float  # median raw library size
    nonzero_frac: float  # fraction of non-zero entries in the model input
    count_p99: float  # 99th pct of model-input expression values


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Cells sampled to estimate a file's median library size in lazy mode (matches
# compute_data_stats' sampling budget). The median is robust, so this is
# sub-percent off the exact value.
_MEDIAN_SAMPLE_CELLS = 20_000


class PerturbationDataset(Dataset):
    """
    Dataset for single-cell perturbation experiments.

    Supports two loading modes controlled by ``config.load_lazily``:

    *Eager* (default): loads the full count matrix into RAM as a dense float32
    array on construction. Fast for datasets that fit in memory (~20 GB).

    *Lazy*: never loads X into RAM. Each DataLoader worker opens its own HDF5
    file handle on the first ``__getitem__`` call after the process is forked,
    which is the correct HDF5 multiprocessing pattern. Preprocessing is applied
    per-row in ``__getitem__``.

    Parameters
    ----------
    config:
        Dataset configuration object. ``config.h5ad_path`` (single file) or
        ``config.h5ad_paths`` (list of files) points to AnnData ``.h5ad`` files
        whose ``adata.X`` is the raw count matrix (dense or CSR sparse). In
        multi-file mode all files must share the same gene set.
    logger:
        Optional logger used for progress/warning messages.
    indices:
        Optional 1-D integer array selecting a subset of cells. Use this to
        create train/val splits that share the same underlying data arrays
        (see ``split_dataset``). When *None* all cells are included.
    """

    def __init__(
        self,
        config: DatasetConfig,
        logger: Any | None = None,
        indices: np.ndarray | None = None,
    ) -> None:
        import anndata
        import scipy.sparse

        self._logger = logger or logging.getLogger(__name__)

        h5ad_paths = _resolve_h5ad_paths(config)
        file_meta = [_resolve_file_metadata(p, config) for p in h5ad_paths]

        t0 = time.perf_counter()

        # --- shared preprocessing state ---
        self._load_lazily: bool = config.load_lazily
        self._h5ad_paths: list[str] = h5ad_paths
        self._file_meta: list[dict[str, Any]] = file_meta
        self._donor_col: str | None = config.donor_col
        # Back-compat single-path field (first file) — used by legacy call sites
        # and kept in sync with _h5ad_paths[0].
        self._h5ad_path: str = h5ad_paths[0]
        self._apply_normalize: bool = config.apply_normalize
        # Normalization target. ``config.target_sum`` may be None → resolved to
        # the per-file median library size at load time. ``_per_file_target_sum``
        # holds the concrete scalar used per file (None when a file is not
        # runtime-normalized — offline-log1p files, or apply_normalize=False).
        # ``_median_library_size`` is the global (dataset-level) median of raw
        # library sizes, used for stats reporting and as the single
        # representative target via the ``normalization_target_sum`` property.
        self._config_target_sum: float | None = config.target_sum
        self._per_file_target_sum: list[float | None] = []
        self._median_library_size: float | None = None
        self._apply_log1p: bool = config.apply_log1p
        self._use_binning: str = config.use_binning

        # --- gene-order permutation state (set during _init_eager / _init_lazy
        # when config.gene_order_path is provided) ---
        self._target_genes: list[str] | None = None
        self._num_hvg: int | None = config.num_hvg
        # adjusted_perm[i] = source col index for target col i, with -1
        # remapped to a sentinel index pointing at an appended zero. One entry
        # per source file.
        self._gene_perms_per_file: list[np.ndarray] | None = None
        self._n_native_per_file: list[int] | None = None
        # HVG-prefix slice of ``_gene_perms_per_file`` plus a per-file
        # "all entries < n_native" flag. Cached at init so the per-row hot
        # path in ``_get_lazy_counts`` can skip the 40k+1-element sentinel
        # scratch buffer when the top-N HVGs are all present in this file.
        self._hvg_adjusted_per_file: list[np.ndarray] | None = None
        self._hvg_no_missing_per_file: list[bool] | None = None
        if config.gene_order_path is not None:
            self._target_genes = [
                line.strip()
                for line in Path(config.gene_order_path).read_text().splitlines()
                if line.strip()
            ]
            if not self._target_genes:
                raise ValueError(f"gene_order_path is empty: {config.gene_order_path}")
            if self._num_hvg is not None and self._num_hvg > len(self._target_genes):
                raise ValueError(
                    f"num_hvg={self._num_hvg} exceeds gene_order length "
                    f"{len(self._target_genes)} in {config.gene_order_path}"
                )

        # --- lazy-mode per-file handles (opened per worker on first __getitem__) ---
        self._lazy_X_per_file: list | None = None
        self._lazy_h5_files: list | None = None
        # Back-compat single-handle field — always None, still referenced by __getstate__.
        self._lazy_X = None

        if self._use_binning not in _BINNING_MODES:
            raise ValueError(
                f"use_binning={self._use_binning!r} is invalid; "
                f"must be one of {_BINNING_MODES}."
            )
        if self._use_binning == "quantile":
            if self._apply_log1p:
                raise ValueError(
                    "use_binning='quantile' requires apply_log1p=False — "
                    "binning operates on integer counts."
                )
            self._log_info(
                f"use_binning='quantile': counts mapped to {COUNT_BIN_NUM_CLASSES} "
                f"bins with upper boundaries {COUNT_BIN_UPPER.tolist()} "
                f"(last bin = lump bucket for counts > {int(COUNT_BIN_UPPER[-1])})."
            )

        self._log_info(
            f"Loading {len(h5ad_paths)} file(s): "
            f"{[m['basename'] for m in file_meta]}"
        )

        if config.load_lazily:
            self._init_lazy(config, anndata)
        else:
            self._init_eager(config, anndata, scipy.sparse)

        # Default to None unless _init_lazy filled them for derive_log1p files.
        # Eager mode never reads them (it materialises log1p at init).
        if not hasattr(self, "_derive_log1p_per_file"):
            self._derive_log1p_per_file = None
            self._lib_full_per_file = None
            self._median_lib_per_file = None

        # --- optional perturbation embeddings ---
        self.pert_emb: torch.Tensor | None = None
        if config.pert_mapping_path is not None:
            self.pert_emb = self._load_pert_emb(config.pert_mapping_path)

        # --- optional subset index (for train/val splits) ---
        self._indices: np.ndarray | None = (
            np.asarray(indices, dtype=np.int64) if indices is not None else None
        )

        elapsed = time.perf_counter() - t0
        self._log_info(
            f"Done in {elapsed:.1f}s.  {len(self):,} cells exposed, "
            f"{self._n_perts} unique perts, {self._n_cell_types} cell types."
        )

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _extract_file_labels(self, obs, meta: dict[str, Any], n_rows: int):
        """
        Return ``(pert_labels, cell_type_labels, donor_labels)`` as string
        arrays of length ``n_rows`` for one file.

        ``donor_labels`` is ``None`` when ``self._donor_col`` is unset; otherwise
        the column must exist on the file's obs (strict — raises ValueError on
        miss so typos are caught at init time).

        Observational files (``meta['pert_col'] is None``) produce a pert_labels
        array filled with ``meta['control_label']``.
        """
        # --- cell type ---
        if "cell_type" in obs.columns:
            cell_type_labels = obs["cell_type"].astype(str).to_numpy()
        else:
            fallback = meta["cell_type_fallback"]
            if fallback is None:
                fallback = os.path.splitext(meta["basename"])[0]
                self._log_warning(
                    f"  [{meta['basename']}] no 'cell_type' obs column and "
                    f"no DATASET_METADATA entry; defaulting cell_type to "
                    f"file stem {fallback!r}."
                )
            cell_type_labels = np.full(n_rows, str(fallback), dtype=object)

        # --- perturbation ---
        pert_col = meta["pert_col"]
        control_label = meta["control_label"]
        if pert_col is None:
            if control_label is None:
                raise ValueError(
                    f"File {meta['basename']!r} is observational "
                    f"(pert_col=None) but control_label is None — cannot "
                    f"synthesise a control label."
                )
            pert_labels = np.full(n_rows, str(control_label), dtype=object)
            self._log_info(
                f"  [{meta['basename']}] observational: all {n_rows:,} cells "
                f"labelled '{control_label}'."
            )
        else:
            if pert_col not in obs.columns:
                raise ValueError(
                    f"File {meta['basename']!r} has no '{pert_col}' obs "
                    f"column (resolved from metadata / DatasetConfig.pert_col)."
                )
            pert_labels = obs[pert_col].astype(str).to_numpy()

        # --- donor (optional, only when DatasetConfig.donor_col is set) ---
        donor_labels: np.ndarray | None = None
        if self._donor_col is not None:
            if self._donor_col not in obs.columns:
                raise ValueError(
                    f"File {meta['basename']!r} has no '{self._donor_col}' obs "
                    f"column (required by DatasetConfig.donor_col)."
                )
            donor_labels = obs[self._donor_col].astype(str).to_numpy()

        return pert_labels, cell_type_labels, donor_labels

    def _build_unified_indices(
        self,
        pert_labels_per_file: list[np.ndarray],
        cell_type_labels_per_file: list[np.ndarray],
        donor_labels_per_file: list[np.ndarray | None] | None = None,
    ) -> None:
        """
        Build the unified (cross-file) perturbation, cell-type, and (optional)
        donor vocabularies and flatten per-file label lists into
        ``_pert_indices`` / ``_cell_type_indices`` / ``_donor_indices``.

        ``donor_labels_per_file`` is ``None`` (or all entries are ``None``) when
        ``DatasetConfig.donor_col`` is unset; in that case ``_donor_indices``
        and ``unique_donors`` are left as ``None`` / empty.
        """
        all_perts = (
            np.concatenate(pert_labels_per_file)
            if pert_labels_per_file
            else np.array([], dtype=object)
        )
        all_cts = (
            np.concatenate(cell_type_labels_per_file)
            if cell_type_labels_per_file
            else np.array([], dtype=object)
        )

        self.unique_perts: list[str] = sorted(set(all_perts.tolist()))
        self.pert_to_idx: dict[str, int] = {
            p: i for i, p in enumerate(self.unique_perts)
        }
        self._pert_indices: np.ndarray = np.array(
            [self.pert_to_idx[p] for p in all_perts], dtype=np.int64
        )
        self._n_perts: int = len(self.unique_perts)

        self.unique_cell_types: list[str] = sorted(set(all_cts.tolist()))
        self.cell_type_to_idx: dict[str, int] = {
            c: i for i, c in enumerate(self.unique_cell_types)
        }
        self._cell_type_indices: np.ndarray = np.array(
            [self.cell_type_to_idx[c] for c in all_cts], dtype=np.int64
        )
        self._n_cell_types: int = len(self.unique_cell_types)

        # --- optional donor vocabulary ---
        self.unique_donors: list[str] = []
        self.donor_to_idx: dict[str, int] = {}
        self._donor_indices: np.ndarray | None = None
        if donor_labels_per_file is not None and any(
            d is not None for d in donor_labels_per_file
        ):
            all_donors = np.concatenate(
                [d for d in donor_labels_per_file if d is not None]
            )
            self.unique_donors = sorted(set(all_donors.tolist()))
            self.donor_to_idx = {d: i for i, d in enumerate(self.unique_donors)}
            self._donor_indices = np.array(
                [self.donor_to_idx[d] for d in all_donors], dtype=np.int64
            )

    def _validate_matching_var(
        self, var_names_ref: np.ndarray, var_names: np.ndarray, basename: str
    ) -> None:
        """Error on differing n_vars; warn if names differ at matching length.

        Different naming conventions (e.g. ENSG IDs vs gene symbols) are
        allowed as long as caller has pre-aligned gene order — we only
        enforce column-count equality.
        """
        if len(var_names_ref) != len(var_names):
            raise ValueError(
                f"Multi-file load: file {basename!r} has {len(var_names)} "
                f"genes but first file has {len(var_names_ref)}. All files "
                f"must share the same gene set. Set ``dataset.gene_order_path`` "
                f"to a canonical gene list to permute/zero-fill each file to a "
                f"common order, or pre-align the files."
            )
        if not np.array_equal(var_names_ref, var_names):
            mismatch = int((var_names_ref != var_names).sum())
            self._log_warning(
                f"  [{basename}] var_names differ from the first file at "
                f"{mismatch}/{len(var_names)} positions. Assuming gene "
                f"order is aligned (different naming conventions are OK)."
            )

    def _build_gene_permutation(
        self, var_names: np.ndarray, basename: str
    ) -> np.ndarray:
        """Build the per-file permutation array mapping target → source columns.

        Returns ``adjusted`` of length ``len(self._target_genes)``. Entry i is
        either the source column index for target gene i, or ``len(var_names)``
        — a sentinel pointing at an appended zero in the extended row buffer.

        If a literal string match leaves the majority of target genes unmatched
        (typical when ``gene_order_path`` is in ENSG and the h5ad's var_names
        are symbols, or vice versa), we re-attempt the lookup using mygene to
        convert IDs in whichever direction the mismatch implies.
        """
        assert self._target_genes is not None
        src_lookup: dict[str, int] = {}
        for i, name in enumerate(var_names):
            # Keep the FIRST occurrence; warn on duplicates.
            src_lookup.setdefault(str(name), i)

        n_native = len(var_names)
        perm = np.fromiter(
            (src_lookup.get(g, -1) for g in self._target_genes),
            dtype=np.int64,
            count=len(self._target_genes),
        )
        n_missing = int((perm < 0).sum())
        n_matched = len(perm) - n_missing

        if n_missing > n_matched:
            extra = self._mygene_fallback(perm, src_lookup, basename)
            n_missing -= extra
            n_matched += extra

        self._log_info(
            f"  [{basename}] gene_order: matched {n_matched}/{len(perm)} "
            f"target genes; {n_missing} zero-filled"
        )
        adjusted = np.where(perm < 0, n_native, perm).astype(np.int64)
        return adjusted

    def _mygene_fallback(
        self, perm: np.ndarray, src_lookup: dict[str, int], basename: str
    ) -> int:
        """In-place fill of unmatched ``perm`` entries via mygene ID conversion.

        Returns the number of newly resolved entries. Modifies ``perm``.
        """
        assert self._target_genes is not None
        unmatched_idx = [i for i, p in enumerate(perm) if p < 0]
        if not unmatched_idx:
            return 0

        unmatched_genes = [self._target_genes[i] for i in unmatched_idx]
        n_ensg = sum(1 for g in unmatched_genes if str(g).startswith("ENSG"))
        targets_are_ensg = n_ensg > len(unmatched_genes) // 2

        try:
            import mygene
        except ImportError:
            self._log_warning(
                f"  [{basename}] gene_order: majority unmatched but `mygene` "
                f"is not installed — skipping ID-conversion fallback."
            )
            return 0

        self._log_info(
            f"  [{basename}] gene_order: only {len(perm) - len(unmatched_idx)}"
            f"/{len(perm)} matched directly — querying mygene "
            f"({'ENSG -> symbol' if targets_are_ensg else 'symbol -> ENSG'}) "
            f"for {len(unmatched_genes)} unmatched"
        )
        try:
            mg = mygene.MyGeneInfo()
            if targets_are_ensg:
                results = mg.querymany(
                    unmatched_genes,
                    scopes="ensembl.gene",
                    fields="symbol",
                    species="human",
                    returnall=False,
                )
                conv: dict[str, str] = {
                    r["query"]: r["symbol"]
                    for r in results
                    if not r.get("notfound") and r.get("symbol")
                }
            else:
                results = mg.querymany(
                    unmatched_genes,
                    scopes="symbol",
                    fields="ensembl.gene",
                    species="human",
                    returnall=False,
                )
                conv = {}
                for r in results:
                    if r.get("notfound"):
                        continue
                    eg = r.get("ensembl")
                    gid = None
                    if isinstance(eg, dict):
                        gid = eg.get("gene")
                    elif isinstance(eg, list) and eg:
                        first = eg[0]
                        gid = first.get("gene") if isinstance(first, dict) else first
                    if gid:
                        conv[r["query"]] = gid
        except Exception as e:
            self._log_warning(
                f"  [{basename}] gene_order: mygene query failed ({e}) — "
                f"falling back to zero-fill for unmatched entries."
            )
            return 0

        n_new = 0
        for i in unmatched_idx:
            converted = conv.get(self._target_genes[i])
            if converted is None:
                continue
            idx = src_lookup.get(converted)
            if idx is None:
                continue
            perm[i] = idx
            n_new += 1

        self._log_info(
            f"  [{basename}] gene_order: mygene resolved {n_new} "
            f"additional / {len(unmatched_genes)} unmatched"
        )
        return n_new

    def _apply_permutation_dense(
        self, part: np.ndarray, adjusted_perm: np.ndarray, n_native: int
    ) -> np.ndarray:
        """Apply the permutation to a dense (n_obs, n_native) array via the
        sentinel-zero trick. Returns shape (n_obs, len(self._target_genes))."""
        # Append a zero column so sentinel index n_native pulls zeros.
        extended = np.empty((part.shape[0], n_native + 1), dtype=part.dtype)
        extended[:, :n_native] = part
        extended[:, n_native] = 0.0
        return extended[:, adjusted_perm]

    def _init_eager(self, config: DatasetConfig, anndata, scipy_sparse) -> None:
        """Load every file's X into RAM, concat along the cell axis."""
        import scanpy as sc

        counts_parts: list[np.ndarray] = []
        library_size_parts: list[np.ndarray] = []
        per_file_target_sum: list[float | None] = []
        pert_labels_per_file: list[np.ndarray] = []
        cell_type_labels_per_file: list[np.ndarray] = []
        donor_labels_per_file: list[np.ndarray | None] = []
        var_names_ref: np.ndarray | None = None
        n_vars: int | None = None
        # When gene_order_path is set, each file is permuted to the canonical
        # order before concat, so cross-file native gene alignment is not required.
        skip_cross_file_validation = self._target_genes is not None

        for path, meta in zip(self._h5ad_paths, self._file_meta):
            adata = anndata.read_h5ad(path)
            n_obs, n_genes = adata.shape
            est_gb = n_obs * n_genes * 4 / 1024**3
            self._log_info(
                f"[{meta['basename']}] Loading "
                f"({n_obs:,} cells × {n_genes:,} genes, dense ≈ {est_gb:.1f} GB) …"
            )
            precomputed_layer = meta.get("precomputed_log1p_layer")
            derive_log1p = meta.get("derive_log1p_via_lib_full", False)
            has_offline_log1p = precomputed_layer is not None or derive_log1p
            if has_offline_log1p:
                # File has normalize_total + log1p baked in offline (cached
                # layer or derivable from raw + library_size_full). Validate
                # combo + file structure, then skip runtime preprocessing.
                if config.apply_normalize != config.apply_log1p:
                    src = (
                        f"precomputed_log1p_layer={precomputed_layer!r}"
                        if precomputed_layer is not None
                        else "derive_log1p_via_lib_full=True"
                    )
                    raise ValueError(
                        f"[{meta['basename']}] declares {src}; only "
                        f"(apply_normalize=True, apply_log1p=True) and "
                        f"(apply_normalize=False, apply_log1p=False) are "
                        f"supported, got (apply_normalize={config.apply_normalize}, "
                        f"apply_log1p={config.apply_log1p})."
                    )
                if (
                    precomputed_layer is not None
                    and precomputed_layer not in adata.layers
                ):
                    raise ValueError(
                        f"[{meta['basename']}] precomputed_log1p_layer="
                        f"{precomputed_layer!r} not found in adata.layers "
                        f"(available: {list(adata.layers.keys())})."
                    )
                if derive_log1p:
                    if "library_size_full" not in adata.obs.columns:
                        raise ValueError(
                            f"[{meta['basename']}] derive_log1p_via_lib_full=True "
                            f"requires obs['library_size_full']."
                        )
                    if (
                        "preprocessing" not in adata.uns
                        or "median_library_size_full" not in adata.uns["preprocessing"]
                    ):
                        raise ValueError(
                            f"[{meta['basename']}] derive_log1p_via_lib_full=True "
                            f"requires uns['preprocessing']['median_library_size_full']."
                        )
                if config.apply_normalize and config.target_sum is not None:
                    self._log_warning(
                        f"  [{meta['basename']}] offline log1p in use; "
                        f"DatasetConfig.target_sum={config.target_sum} is IGNORED — "
                        f"the file's normalization (median library size) is baked in."
                    )
                # Pre-sliced files (e.g. 2k HVGs) cannot meaningfully apply
                # min_genes / min_cells thresholds tuned for the full panel.
                src_msg = (
                    f"precomputed_log1p_layer={precomputed_layer!r}"
                    if precomputed_layer is not None
                    else "derive_log1p_via_lib_full=True"
                )
                self._log_info(
                    f"  preprocessing: {src_msg} "
                    f"(skip filter_cells, filter_genes, normalize_total, log1p)"
                )
            else:
                self._log_info(
                    "  preprocessing: "
                    f"filter_cells(min_genes={config.min_genes}), "
                    f"filter_genes(min_cells={config.min_cells}), "
                    f"normalize_total={config.apply_normalize}(target_sum={config.target_sum}), "
                    f"log1p={config.apply_log1p}"
                    + (
                        "  [counts will be quantized to nearest int]"
                        if config.apply_normalize and not config.apply_log1p
                        else ""
                    )
                )
            if not has_offline_log1p:
                sc.pp.filter_cells(adata, min_genes=config.min_genes)
            # Apply obs_filters BEFORE filter_genes so per-gene cell-count
            # statistics reflect the filtered cell population.
            obs_filter_mask = _build_obs_filter_mask(
                adata.obs,
                meta["obs_filters"],
                meta["obs_filter_strict_cols"],
                meta["basename"],
                self._logger,
            )
            n_dropped_by_obs = int((~obs_filter_mask).sum())
            if n_dropped_by_obs:
                self._log_info(
                    f"  obs_filters {meta['obs_filters']}: dropped "
                    f"{n_dropped_by_obs:,} cells, kept {int(obs_filter_mask.sum()):,}"
                )
                adata = adata[obs_filter_mask].copy()
            if not has_offline_log1p:
                sc.pp.filter_genes(adata, min_cells=config.min_cells)
            # Capture the per-cell full-panel library size for NB heads /
            # metric-space conversion at eval time. With offline log1p
            # (cached layer or derived), X is the post-slice raw counts so
            # X.sum would only see the post-slice panel — prefer the saved
            # full-panel library size from obs.
            if has_offline_log1p and "library_size_full" in adata.obs.columns:
                file_lib = adata.obs["library_size_full"].to_numpy().astype(np.float32)
            else:
                file_lib = np.asarray(adata.X.sum(axis=1)).ravel().astype(np.float32)
            library_size_parts.append(file_lib)
            # Per-file normalization target: config.target_sum if set, else this
            # file's median library size (scanpy normalize_total(None) semantics).
            # None for files we don't runtime-normalize (offline log1p, or
            # apply_normalize=False) so downstream code skips them.
            file_target_sum: float | None = None
            if not has_offline_log1p:
                if config.apply_normalize:
                    file_target_sum = self._resolve_file_target_sum(file_lib)
                    sc.pp.normalize_total(adata, target_sum=file_target_sum)
                if config.apply_log1p:
                    sc.pp.log1p(adata)
            per_file_target_sum.append(file_target_sum)
            self._log_info(
                f"  post-preprocess: {adata.n_obs:,} cells × {adata.n_vars:,} genes"
            )

            cur_var_names = adata.var_names.to_numpy()
            if not skip_cross_file_validation:
                # Default behavior: enforce native gene alignment across files
                # (filter_genes may drop different genes in different files).
                if var_names_ref is None:
                    var_names_ref = cur_var_names
                    n_vars = adata.n_vars
                else:
                    self._validate_matching_var(
                        var_names_ref, cur_var_names, meta["basename"]
                    )

            # Extract X as dense float32. With offline log1p + apply_log1p=True
            # we either read from the cached layer or derive log1p on-the-fly
            # from raw counts using the file's library_size_full.
            if has_offline_log1p and config.apply_log1p:
                if precomputed_layer is not None:
                    X = adata.layers[precomputed_layer]
                else:
                    median_lib = float(
                        adata.uns["preprocessing"]["median_library_size_full"]
                    )
                    scale = (median_lib / file_lib).astype(np.float32)
                    X_raw = adata.X
                    if scipy_sparse.issparse(X_raw):
                        X_norm = scipy_sparse.diags(scale, format="csr").dot(X_raw)
                        X_norm.data = np.log1p(X_norm.data, dtype=np.float32)
                        X = X_norm
                    else:
                        X = np.log1p(
                            np.asarray(X_raw, dtype=np.float32) * scale[:, None]
                        )
            else:
                X = adata.X
            if scipy_sparse.issparse(X):
                part = np.asarray(X.todense(), dtype=np.float32)
            else:
                part = np.asarray(X, dtype=np.float32)
            # Free obs / labels into local refs so we can drop adata (and its
            # sparse X / layers) before the dense `part` lives the rest of
            # the loop. Saves ~1.5–3 GB transient peak per file.
            obs_local = adata.obs
            n_obs_local = adata.n_obs
            del X, adata
            import gc

            gc.collect()
            if (
                not has_offline_log1p
                and config.apply_normalize
                and not config.apply_log1p
            ):
                part = np.rint(part).astype(np.float32)
            if self._use_binning != "none":
                part = _apply_binning(part, self._use_binning)

            # Apply per-file permutation to the canonical gene order, if set.
            # This happens AFTER preprocessing so normalize_total uses the
            # native (full) panel for depth correction.
            if self._target_genes is not None:
                adjusted = self._build_gene_permutation(cur_var_names, meta["basename"])
                part = self._apply_permutation_dense(
                    part, adjusted, n_native=cur_var_names.shape[0]
                )

            counts_parts.append(part)

            p_labels, ct_labels, d_labels = self._extract_file_labels(
                obs_local, meta, n_obs_local
            )
            pert_labels_per_file.append(p_labels)
            cell_type_labels_per_file.append(ct_labels)
            donor_labels_per_file.append(d_labels)

        # Concat cells across files.
        self.counts: np.ndarray | None = (
            counts_parts[0]
            if len(counts_parts) == 1
            else np.concatenate(counts_parts, axis=0)
        )
        # Per-cell full-panel library size (one float32 per cell). Captured
        # before normalize_total above; used at eval time for CP10K and at
        # train time for any NB head that wants the true library size.
        self._library_size: np.ndarray | None = (
            library_size_parts[0]
            if len(library_size_parts) == 1
            else np.concatenate(library_size_parts, axis=0)
        )
        # Per-file normalization targets + global median (raw library sizes).
        self._per_file_target_sum = per_file_target_sum
        _pos_lib = self._library_size[self._library_size > 0]
        self._median_library_size = float(np.median(_pos_lib)) if _pos_lib.size else 1.0

        # Slice to top num_hvg columns when configured.
        if self._num_hvg is not None:
            self.counts = self.counts[:, : self._num_hvg]

        binned_max = _binning_count_max(self._use_binning)
        if binned_max is not None:
            self._count_max: int | None = binned_max
        elif not config.apply_log1p:
            self._count_max = int(self.counts.max())
        else:
            self._count_max = None

        self._n_obs: int = self.counts.shape[0]
        if self._target_genes is not None:
            self._n_vars: int = (
                self._num_hvg if self._num_hvg is not None else len(self._target_genes)
            )
        else:
            self._n_vars = n_vars or 0

        # Per-file row offsets (cumulative): file_i contributes rows
        # [offsets[i], offsets[i+1]) in the concatenated matrix.
        file_sizes = [p.shape[0] for p in counts_parts]
        self._file_row_offsets: np.ndarray = np.concatenate(
            [[0], np.cumsum(file_sizes)]
        ).astype(np.int64)
        # In eager mode, physical HDF5 indices are not used.
        self._obs_int_indices_per_file: list[np.ndarray] | None = None
        self._obs_int_indices: np.ndarray | None = None

        self._build_unified_indices(
            pert_labels_per_file, cell_type_labels_per_file, donor_labels_per_file
        )

    def _init_lazy(self, config: DatasetConfig, anndata) -> None:
        """
        Open every backed h5ad to read obs only; never load X into RAM. Each
        DataLoader worker later opens its own per-file HDF5 handles on the
        first ``__getitem__`` call (fork-safe).
        """
        pert_labels_per_file: list[np.ndarray] = []
        cell_type_labels_per_file: list[np.ndarray] = []
        donor_labels_per_file: list[np.ndarray | None] = []
        obs_int_indices_per_file: list[np.ndarray] = []
        n_kept_per_file: list[int] = []
        per_file_target_sum: list[float | None] = []
        var_names_ref: np.ndarray | None = None
        n_genes_ref: int | None = None
        # When gene_order_path is set, files are permuted at row-read time, so
        # native gene panels are allowed to differ across files.
        gene_perms_per_file: list[np.ndarray] = []
        n_native_per_file: list[int] = []

        # Per-file caches for derive_log1p_via_lib_full files (lazy mode).
        # Indexed in the same order as self._obs_int_indices_per_file.
        self._derive_log1p_per_file: list[bool] = []
        self._lib_full_per_file: list[np.ndarray | None] = []
        self._median_lib_per_file: list[float | None] = []

        for path, meta in zip(self._h5ad_paths, self._file_meta):
            if meta.get("precomputed_log1p_layer") is not None:
                raise ValueError(
                    f"[{meta['basename']}] declares precomputed_log1p_layer; "
                    "load_lazily=True is not supported for such files. Use "
                    "load_lazily=False, or migrate the file to "
                    "derive_log1p_via_lib_full=True (raw counts only; log1p "
                    "is computed at read time)."
                )
            derive_log1p = bool(meta.get("derive_log1p_via_lib_full", False))
            self._log_info(f"[{meta['basename']}] Opening in lazy (backed='r') mode …")
            adata = anndata.read_h5ad(path, backed="r")
            n_obs_raw, n_genes = adata.shape
            est_gb = n_obs_raw * n_genes * 4 / 1024**3
            self._log_info(
                f"  {n_obs_raw:,} cells × {n_genes:,} genes  (dense ≈ {est_gb:.1f} GB)"
            )

            cur_var_names = adata.var_names.to_numpy()
            if self._target_genes is not None:
                adjusted = self._build_gene_permutation(cur_var_names, meta["basename"])
                gene_perms_per_file.append(adjusted)
                n_native_per_file.append(int(cur_var_names.shape[0]))
            else:
                # Default behavior: enforce native gene alignment across files.
                if var_names_ref is None:
                    var_names_ref = cur_var_names
                    n_genes_ref = n_genes
                else:
                    self._validate_matching_var(
                        var_names_ref, cur_var_names, meta["basename"]
                    )

            # --- filter_cells: use obs['n_genes'] when present; else skip ---
            if "n_genes" in adata.obs.columns:
                keep_mask = adata.obs["n_genes"].to_numpy() >= config.min_genes
            else:
                self._log_warning(
                    f"  load_lazily=True: obs has no 'n_genes' column — "
                    f"filter_cells(min_genes={config.min_genes}) skipped."
                )
                keep_mask = np.ones(n_obs_raw, dtype=bool)

            # AND with obs_filters mask (per-file defaults + YAML overrides).
            obs_filter_mask = _build_obs_filter_mask(
                adata.obs,
                meta["obs_filters"],
                meta["obs_filter_strict_cols"],
                meta["basename"],
                self._logger,
            )
            keep_mask = keep_mask & obs_filter_mask

            n_dropped = int((~keep_mask).sum())
            if n_dropped:
                self._log_info(
                    f"  filter_cells + obs_filters: dropping {n_dropped:,} cells → "
                    f"{int(keep_mask.sum()):,} remain"
                )
            phys = np.where(keep_mask)[0].astype(np.int64)
            obs_filtered = adata.obs.iloc[phys]

            # filter_genes is never applied in lazy mode (requires full X scan).
            self._log_warning(
                f"  load_lazily=True: filter_genes(min_cells={config.min_cells}) skipped "
                f"(requires a full X pass — pre-filter the file if needed)."
            )

            obs_int_indices_per_file.append(phys)
            n_kept_per_file.append(len(phys))

            p_labels, ct_labels, d_labels = self._extract_file_labels(
                obs_filtered, meta, len(phys)
            )
            pert_labels_per_file.append(p_labels)
            cell_type_labels_per_file.append(ct_labels)
            donor_labels_per_file.append(d_labels)

            self._derive_log1p_per_file.append(derive_log1p)
            if derive_log1p:
                if "library_size_full" not in adata.obs.columns:
                    raise ValueError(
                        f"[{meta['basename']}] derive_log1p_via_lib_full=True "
                        f"but obs['library_size_full'] is missing."
                    )
                if (
                    "preprocessing" not in adata.uns
                    or "median_library_size_full" not in adata.uns["preprocessing"]
                ):
                    raise ValueError(
                        f"[{meta['basename']}] derive_log1p_via_lib_full=True but "
                        f"uns['preprocessing']['median_library_size_full'] is missing."
                    )
                lib_full = (
                    obs_filtered["library_size_full"].to_numpy().astype(np.float32)
                )
                median_lib = float(
                    adata.uns["preprocessing"]["median_library_size_full"]
                )
                self._lib_full_per_file.append(lib_full)
                self._median_lib_per_file.append(median_lib)
            else:
                self._lib_full_per_file.append(None)
                self._median_lib_per_file.append(None)

            # Per-file runtime-normalization target (non-offline files only).
            #   offline (derive_log1p) / apply_normalize=False → None (unused).
            #   config.target_sum set                          → that value.
            #   config.target_sum None                         → per-file median
            #     of raw library sizes, estimated from a seeded ≤20k-cell sample
            #     (same cost profile as compute_data_stats; the median is a
            #     robust statistic so the sample is sub-percent off the exact
            #     value). Only the released-dataset-uncommon raw+normalize+None
            #     case reaches here. Eager uses the exact median (it already has
            #     every cell's library in RAM); a given dataset is loaded one way
            #     or the other, so the two never need to agree bit-for-bit.
            if derive_log1p or not config.apply_normalize:
                per_file_target_sum.append(None)
            elif config.target_sum is not None:
                per_file_target_sum.append(float(config.target_sum))
            else:
                sample_phys = phys
                if phys.size > _MEDIAN_SAMPLE_CELLS:
                    rng = np.random.default_rng(0)
                    sample_phys = rng.choice(
                        phys,
                        size=_MEDIAN_SAMPLE_CELLS,
                        replace=False,
                    )
                file_lib = self._backed_row_sums(adata.X, sample_phys)
                per_file_target_sum.append(self._median_positive_library(file_lib))

            adata.file.close()

        self.counts = None
        self._per_file_target_sum = per_file_target_sum
        # Lazy mode: per-cell library size is computed on-demand inside
        # _get_lazy_counts (we already read the full row anyway). No
        # init-time precompute — keeps init fast for huge datasets. The global
        # median library size is filled in lazily by compute_data_stats /
        # the normalization_target_sum property.
        self._library_size: np.ndarray | None = None
        self._count_max = _binning_count_max(self._use_binning)
        self._n_obs: int = int(sum(n_kept_per_file))
        if self._target_genes is not None:
            self._gene_perms_per_file = gene_perms_per_file
            self._n_native_per_file = n_native_per_file
            self._n_vars: int = (
                self._num_hvg if self._num_hvg is not None else len(self._target_genes)
            )
            # HVG fast-path cache: pre-slice adjusted to the top-N HVG prefix
            # and check whether every entry is < n_native (i.e. no target gene
            # is missing from this file's native panel). When that's true, the
            # per-row hot path can skip the sentinel-extended scratch buffer.
            if self._num_hvg is not None:
                self._hvg_adjusted_per_file = [
                    np.ascontiguousarray(adj[: self._num_hvg])
                    for adj in self._gene_perms_per_file
                ]
                self._hvg_no_missing_per_file = [
                    bool((adj < n_native).all())
                    for adj, n_native in zip(
                        self._hvg_adjusted_per_file, self._n_native_per_file
                    )
                ]
        else:
            self._n_vars = n_genes_ref or 0

        self._file_row_offsets: np.ndarray = np.concatenate(
            [[0], np.cumsum(n_kept_per_file)]
        ).astype(np.int64)
        self._obs_int_indices_per_file: list[np.ndarray] | None = (
            obs_int_indices_per_file
        )
        # Legacy single-file alias (first file). Unused in multi-file mode.
        self._obs_int_indices: np.ndarray | None = obs_int_indices_per_file[0]

        self._build_unified_indices(
            pert_labels_per_file, cell_type_labels_per_file, donor_labels_per_file
        )

        self._log_info(
            f"  Lazy init complete: {self._n_obs:,} cells × {self._n_vars:,} genes "
            f"across {len(self._h5ad_paths)} file(s)."
        )

    def _locate_global_row(self, global_idx: int) -> tuple[int, int]:
        """Map a global cell index to (file_idx, local_row_within_file)."""
        offsets = self._file_row_offsets
        # offsets has shape (n_files + 1,); the file index is
        # searchsorted(offsets[1:], global_idx, side='right').
        if len(offsets) <= 2:
            # Single-file fast path.
            return 0, global_idx
        file_idx = int(np.searchsorted(offsets[1:], global_idx, side="right"))
        return file_idx, int(global_idx - offsets[file_idx])

    # ------------------------------------------------------------------
    # Pickle support (for DataLoader multiprocessing)
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Strip per-process HDF5 handles — workers open their own after fork.
        state["_lazy_X"] = None
        state["_lazy_X_per_file"] = None
        state["_lazy_h5_files"] = None
        # Legacy fields, present in pickles from older versions.
        state.pop("_lazy_adata_per_file", None)
        state.pop("_lazy_adata", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return self._n_obs

    def __getitem__(self, idx: int) -> dict[str, Any]:
        real_idx = int(self._indices[idx]) if self._indices is not None else idx
        pert_idx = int(self._pert_indices[real_idx])
        cell_type_idx = int(self._cell_type_indices[real_idx])

        if self._load_lazily:
            counts_tensor, library_size = self._get_lazy_counts(real_idx)
        else:
            counts_tensor = torch.from_numpy(self.counts[real_idx])
            # Eager mode: precomputed at init time before normalize_total.
            library_size = (
                float(self._library_size[real_idx])
                if self._library_size is not None
                else float(counts_tensor.sum().item())
            )

        out: dict[str, Any] = {
            "counts": counts_tensor,
            "library_size": library_size,
            "pert_idx": pert_idx,
            "cell_type_idx": cell_type_idx,
            "cell_type": self.unique_cell_types[cell_type_idx],
        }
        if self.pert_emb is not None:
            out["pert_emb"] = self.pert_emb[pert_idx]
        return out

    def _open_lazy_handles(self) -> None:
        """Open one HDF5 handle per source file (fork-safe, lazy).

        Bypasses ``anndata.read_h5ad(backed='r')`` because that call eagerly
        materialises the full obs DataFrame (multiple GB on large atlases).
        Workers only need to slice X rows — obs is already cached on the
        parent as ``_pert_indices`` / ``_cell_type_indices`` numpy arrays —
        so we open the h5py file directly and wrap X with anndata's
        ``sparse_dataset``. With ``num_workers=4`` this saves ~12 GB host RAM.
        """
        import h5py
        from anndata.io import sparse_dataset

        self._lazy_X_per_file = []
        self._lazy_h5_files = []
        for path in self._h5ad_paths:
            f = h5py.File(path, "r")
            x_node = f["X"]
            x = sparse_dataset(x_node) if isinstance(x_node, h5py.Group) else x_node
            self._lazy_X_per_file.append(x)
            # Keep the h5py.File reference so its dataset handles stay open.
            self._lazy_h5_files.append(f)
        self._lazy_X = self._lazy_X_per_file[0]

    def _get_lazy_counts(
        self,
        global_idx: int,
    ) -> tuple[torch.Tensor, float]:
        """
        Read one cell from the correct file's HDF5 handle, apply preprocessing,
        return ``(counts_tensor, library_size)`` where ``library_size`` is the
        full-panel pre-normalize row sum (used for NB / CP10K eval).

        Opens all per-file handles on the first call in each process
        (fork-safe: handles are created after the DataLoader worker is forked).
        """
        import scipy.sparse

        if self._lazy_X_per_file is None:
            self._open_lazy_handles()

        file_idx, local_idx = self._locate_global_row(global_idx)
        phys_idx = int(self._obs_int_indices_per_file[file_idx][local_idx])
        row = self._lazy_X_per_file[file_idx][phys_idx]
        if scipy.sparse.issparse(row):
            row = row.toarray().ravel()
        counts_row = np.asarray(row, dtype=np.float32).ravel()

        # Files with derive_log1p_via_lib_full=True hold raw counts on a
        # pre-sliced HVG panel; library_size and the normalization scale
        # come from the cached full-panel library_size_full (saved at init
        # from obs), NOT the post-slice row sum.
        if (
            self._derive_log1p_per_file is not None
            and self._derive_log1p_per_file[file_idx]
        ):
            library_size = float(self._lib_full_per_file[file_idx][local_idx])
            median_lib = self._median_lib_per_file[file_idx]
            if self._apply_log1p:
                if library_size > 0:
                    counts_row = np.log1p(
                        counts_row * (median_lib / library_size), dtype=np.float32
                    )
                else:
                    counts_row = np.log1p(counts_row, dtype=np.float32)
            # apply_log1p=False → return raw counts unchanged.
        else:
            # Capture full-panel library size BEFORE any preprocessing /
            # HVG slicing. We already paid the I/O for the row, so the sum
            # is essentially free.
            library_size = float(counts_row.sum())

            if self._apply_normalize:
                if library_size > 0:
                    counts_row = counts_row * (
                        self._per_file_target_sum[file_idx] / library_size
                    )
                if not self._apply_log1p:
                    counts_row = np.rint(counts_row)
            if self._apply_log1p:
                counts_row = np.log1p(counts_row)
        if self._use_binning != "none":
            counts_row = _apply_binning(counts_row, self._use_binning)

        # Permute to canonical gene order (and optionally slice top-N HVGs).
        if self._gene_perms_per_file is not None:
            if (
                self._hvg_adjusted_per_file is not None
                and self._hvg_no_missing_per_file[file_idx]
            ):
                # Fast path: every top-N HVG exists in this file's native
                # panel, so we can fancy-index ``counts_row`` directly into
                # the (num_hvg,) output. Skips the ~n_native-sized scratch
                # buffer that the sentinel-zero trick would otherwise need.
                counts_row = counts_row[self._hvg_adjusted_per_file[file_idx]]
            else:
                # Sentinel-zero trick: append a single 0 to the row, then
                # fancy-index using a precomputed array where missing target
                # genes point at the appended sentinel.
                n_native = self._n_native_per_file[file_idx]
                if self._hvg_adjusted_per_file is not None:
                    adjusted = self._hvg_adjusted_per_file[file_idx]
                else:
                    adjusted = self._gene_perms_per_file[file_idx]
                extended = np.empty(n_native + 1, dtype=np.float32)
                extended[:n_native] = counts_row
                extended[n_native] = 0.0
                counts_row = extended[adjusted]

        # .copy() ensures the array is writeable (h5py may return read-only views)
        return torch.from_numpy(counts_row.copy()), library_size

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_genes(self) -> int:
        return self._n_vars

    @property
    def num_perts(self) -> int:
        return self._n_perts

    @property
    def num_cell_types(self) -> int:
        return self._n_cell_types

    @property
    def num_cells(self) -> int:
        return len(self)

    @property
    def count_max(self) -> int | None:
        """Maximum integer count in the dataset. None when data is log-normalised or lazy."""
        return self._count_max

    @property
    def pert_emb_dim(self) -> int | None:
        """Perturbation embedding dimension. None when no pert_mapping was loaded."""
        return self.pert_emb.shape[-1] if self.pert_emb is not None else None

    @property
    def normalization_target_sum(self) -> float:
        """Single representative normalization target for the whole dataset.

        Returns ``config.target_sum`` when set, otherwise the global median of
        raw per-cell library sizes. Used where one scalar must stand in for the
        dataset (the DEG-table builder's metric-space match, the rare
        normalize+log1p=False count_max scan). The actual per-cell data
        normalization uses the *per-file* targets in ``_per_file_target_sum``;
        for single-file datasets the two coincide exactly.
        """
        if self._config_target_sum is not None:
            return float(self._config_target_sum)
        if self._median_library_size is None:
            self._median_library_size = self._resolve_global_median_library_size()
        return float(self._median_library_size)

    def _resolve_global_median_library_size(self) -> float:
        """Global median of raw positive library sizes over this view's cells.

        Eager: exact (from the cached ``_library_size``). Lazy: estimated from a
        20k-cell sample (cheap; only the diagnostic / representative-target
        paths use it). Returns 1.0 as a degenerate fallback.
        """
        if self._library_size is not None and len(self._library_size):
            libs = (
                self._library_size[self._indices]
                if self._indices is not None
                else self._library_size
            )
            pos = libs[libs > 0]
            return float(np.median(pos)) if pos.size else 1.0
        # Lazy fallback: sample raw library sizes via a temporary backed handle.
        import anndata
        import scipy.sparse

        n_cells = len(self)
        indices = (
            self._indices
            if self._indices is not None
            else np.arange(n_cells, dtype=np.int64)
        )
        rng = np.random.default_rng(0)
        sample = indices[rng.choice(n_cells, size=min(n_cells, 20_000), replace=False)]
        tmp = [anndata.read_h5ad(p, backed="r") for p in self._h5ad_paths]
        try:
            libs = self._depth_sparse_sample(
                [a.X for a in tmp],
                sample,
                self._file_row_offsets,
                4096,
                scipy.sparse,
            )
        finally:
            for a in tmp:
                a.file.close()
        pos = libs[libs > 0]
        return float(np.median(pos)) if pos.size else 1.0

    @staticmethod
    def _median_positive_library(file_lib: np.ndarray) -> float:
        """Median of positive entries (matches scanpy normalize_total(None))."""
        pos = file_lib[file_lib > 0]
        return float(np.median(pos)) if pos.size else 1.0

    def _resolve_file_target_sum(self, file_lib: np.ndarray) -> float:
        """Per-file normalization target: config value, or the file's median lib."""
        if self._config_target_sum is not None:
            return float(self._config_target_sum)
        return self._median_positive_library(file_lib)

    # ------------------------------------------------------------------
    # Dataset statistics
    # ------------------------------------------------------------------

    def compute_data_stats(self, n_sample_cells: int = 20_000) -> DataStats:
        """
        Compute summary statistics over the cells exposed by this dataset.

        Respects ``_indices`` so it is safe to call on train/val splits.

        In lazy mode, a fresh temporary HDF5 handle is opened (separate from the
        per-worker ``_lazy_X`` handles). Depth is computed in sorted chunks for
        HDF5 read efficiency. Expression stats are estimated from a random sample
        of ``n_sample_cells`` rows with preprocessing applied, so they reflect
        what the model actually receives.
        """
        indices = self._indices
        n_cells = len(self)
        _CHUNK = 4096

        if self._load_lazily:
            return self._compute_data_stats_lazy(
                n_cells, indices, n_sample_cells, _CHUNK
            )

        # --- eager path ---
        # Depth / library size MUST come from the RAW per-cell library size
        # (captured before normalize_total), NOT from ``self.counts`` — which
        # by this point holds normalized / log1p'd values (or quantile bin IDs).
        # ``self._library_size`` is the full-panel raw count sum per cell.
        depth = (
            self._library_size[indices] if indices is not None else self._library_size
        ).astype(np.float64)
        log_depth = np.log(np.maximum(depth, 1.0))
        _pos = depth[depth > 0]
        self._median_library_size = float(np.median(_pos)) if _pos.size else 1.0

        # Expression-value stats use ``self.counts`` (the actual model input) —
        # its sparsity / magnitude is what we want to report there.
        sample_size = min(n_cells, n_sample_cells)
        rng = np.random.default_rng(0)
        sample_pos = rng.choice(n_cells, size=sample_size, replace=False)
        sample_rows = indices[sample_pos] if indices is not None else sample_pos
        flat = self.counts[sample_rows].ravel()

        return self._make_data_stats(n_cells, depth, log_depth, flat)

    def _compute_data_stats_lazy(
        self,
        n_cells: int,
        indices: np.ndarray | None,
        n_sample_cells: int,
        chunk_size: int,
    ) -> DataStats:
        """Lazy-mode dataset statistics.

        Three independent passes, each as cheap as it can be:

        1. **count_max** — fast paths first.
           * Binning mode → fixed by the bin scheme (no I/O).
           * ``apply_log1p=True`` → count_max unused downstream (no I/O).
           * Raw integer counts (``log1p=False, normalize=False, binning=none``)
             → sequential ``max(X.data)`` per file. O(nnz) sequential I/O,
             no random row reads, no dense conversion. May read data values
             from cells dropped by filter_cells/obs_filters — that only
             over-estimates count_max, which safely sizes embeddings bigger.
           * ``normalize=True + log1p=False`` (rare) → sample-based row scan.
        2. **depth percentiles** — uniform sample of ``n_sample_cells``,
           sparse ``chunk.sum(axis=1)`` (no densify, no permute, no HVG slice
           — sums are column-order-invariant).
        3. **expression-value percentiles** — same uniform sample, but
           densified + permuted + HVG-sliced + preprocessed so values match
           what the model sees.
        """
        import anndata
        import scipy.sparse
        from tqdm.auto import tqdm as _tqdm

        # Per-file backed handles (main-process safe; independent of the
        # per-worker _lazy_X_per_file handles).
        tmp_adatas = [anndata.read_h5ad(p, backed="r") for p in self._h5ad_paths]
        X_per_file = [a.X for a in tmp_adatas]
        offsets = self._file_row_offsets

        # Resolve the global-index → (file_idx, phys_idx) mapping for this split.
        if indices is not None:
            global_idx = indices
        else:
            global_idx = np.arange(n_cells, dtype=np.int64)

        # Uniform cell sample for both depth and expression-value stats.
        sample_size = min(n_cells, n_sample_cells)
        rng = np.random.default_rng(0)
        sample_global = global_idx[rng.choice(n_cells, size=sample_size, replace=False)]

        # ------------------------------------------------------------------
        # 1) count_max
        # ------------------------------------------------------------------
        binned_max = _binning_count_max(self._use_binning)
        if binned_max is not None:
            self._count_max = binned_max
        elif self._apply_log1p:
            # _count_max stays None (set in _init_lazy) — unused downstream.
            pass
        elif not self._apply_normalize:
            # Raw integer counts: sequential scan of CSR data arrays.
            self._count_max = int(self._scan_count_max_raw_sequential(X_per_file))
            self._log_info(f"count_max (sequential X.data scan): {self._count_max}")
        else:
            # Rare: normalize=True + log1p=False. count_max depends on per-row
            # normalization, so we can't shortcut via X.data — scan the sample.
            self._log_warning(
                "count_max under apply_normalize=True + apply_log1p=False is "
                "estimated from a sample; may underestimate population max."
            )
            self._count_max = int(
                self._scan_count_max_normalize_sample(
                    X_per_file,
                    sample_global,
                    offsets,
                    chunk_size,
                )
            )

        # ------------------------------------------------------------------
        # 2) Library size (depth) — raw row sums on the sample, no densify
        # ------------------------------------------------------------------
        depth = self._depth_sparse_sample(
            X_per_file,
            sample_global,
            offsets,
            chunk_size,
            scipy.sparse,
        )
        log_depth = np.log(np.maximum(depth, 1.0))
        # Global median of raw library sizes (estimated from the sample). Also
        # backs normalization_target_sum for the lazy representative target.
        _pos = depth[depth > 0]
        self._median_library_size = float(np.median(_pos)) if _pos.size else 1.0

        # ------------------------------------------------------------------
        # 3) Expression-value stats — densified + preprocessed sample
        # ------------------------------------------------------------------
        sample, sample_full_row_sums = self._read_dense_sample_aligned(
            X_per_file,
            sample_global,
            offsets,
            scipy.sparse,
        )
        if self._apply_normalize:
            # Use the FULL-panel row sums (computed pre-HVG-slice). Using
            # ``sample.sum(axis=1)`` here would divide by HVG-only library,
            # inflating CP-N. Diagnostic stat → the dataset's representative
            # target (config value, or global median) stands in for the
            # per-file targets used by the actual per-cell normalization.
            row_sums = np.where(
                sample_full_row_sums == 0,
                1.0,
                sample_full_row_sums,
            )[:, None]
            sample = sample / row_sums * self.normalization_target_sum
            if not self._apply_log1p:
                sample = np.rint(sample)
        if self._apply_log1p:
            sample = np.log1p(sample)
        if self._use_binning != "none":
            sample = _apply_binning(sample, self._use_binning)
        flat = sample.ravel()

        for a in tmp_adatas:
            a.file.close()

        return self._make_data_stats(n_cells, depth, log_depth, flat)

    # ------------------------------------------------------------------
    # _compute_data_stats_lazy helpers
    # ------------------------------------------------------------------

    def _scan_count_max_raw_sequential(
        self,
        X_per_file: list,
        block_size: int = 1_000_000,
    ) -> float:
        """Global max over CSR ``data`` arrays — sequential I/O, no row scan.

        For backed CSR (the typical lazy case) ``X.data`` is an h5py-backed
        ndarray of stored nonzero values. Reading it in fixed-size blocks is
        sequential I/O on a single dataset — no seeks, no dense conversion,
        no row reconstruction. ~1000× faster than the row-iteration path on
        Parse-scale (~10M cells × 40k genes).
        """
        overall_max = 0.0
        for X in X_per_file:
            data = getattr(X, "data", None)
            if data is None:
                # anndata's modern `_CSRDataset` doesn't expose `.data` as an
                # attribute — the lazy h5py array lives at X.group["data"].
                grp = getattr(X, "group", None) or getattr(X, "_group", None)
                if grp is not None and "data" in grp:
                    data = grp["data"]
            if data is None:
                # Non-sparse fallback: full array (rare for real datasets).
                arr = np.asarray(X[:])
                if arr.size:
                    overall_max = max(overall_max, float(arr.max()))
                continue
            n = int(data.shape[0])
            for start in range(0, n, block_size):
                block = np.asarray(data[start : start + block_size])
                if block.size:
                    overall_max = max(overall_max, float(block.max()))
        return overall_max

    def _scan_count_max_normalize_sample(
        self,
        X_per_file,
        sample_global,
        offsets,
        chunk_size,
    ) -> float:
        """Sample-based count_max for ``normalize=True + log1p=False``.

        This regime requires per-row normalization (``rint(row * target_sum
        / row.sum())``), so the sparse-data shortcut doesn't apply. Scans
        the cell sample only — may under-estimate the population max.
        """
        import scipy.sparse as sp
        from tqdm.auto import tqdm as _tqdm

        sample_file_ids = np.searchsorted(offsets[1:], sample_global, side="right")
        sample_local = sample_global - offsets[sample_file_ids]
        plan = []
        for f_idx in range(len(X_per_file)):
            rows = sample_local[sample_file_ids == f_idx]
            if rows.size:
                phys = self._obs_int_indices_per_file[f_idx][rows]
                plan.append((f_idx, phys))
        total_chunks = sum((p.size + chunk_size - 1) // chunk_size for _, p in plan)

        overall_max = 0.0
        pbar = _tqdm(
            total=total_chunks, desc="count_max sample (normalize+log1p=False)"
        )
        for f_idx, phys in plan:
            for start in range(0, phys.size, chunk_size):
                chunk_phys = np.sort(phys[start : start + chunk_size])
                chunk = X_per_file[f_idx][chunk_phys]
                if sp.issparse(chunk):
                    chunk = chunk.toarray()
                chunk = np.asarray(chunk, dtype=np.float32)
                row_sums = chunk.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1.0, row_sums)
                chunk_max = float(
                    np.rint(chunk / row_sums * self.normalization_target_sum).max()
                )
                if chunk_max > overall_max:
                    overall_max = chunk_max
                pbar.update(1)
        pbar.close()
        return overall_max

    def _backed_row_sums(
        self,
        X,
        phys: np.ndarray,
        chunk_size: int = 4096,
    ) -> np.ndarray:
        """Per-row raw library sizes for ``phys`` rows of a (backed) matrix.

        Reads in sorted chunks for HDF5 efficiency; no densification of the
        whole panel. Used at lazy init to compute a file's median library size.
        """
        import scipy.sparse as sp

        parts: list[np.ndarray] = []
        for start in range(0, phys.size, chunk_size):
            chunk_phys = np.sort(phys[start : start + chunk_size])
            chunk = X[chunk_phys]
            if sp.issparse(chunk):
                parts.append(np.asarray(chunk.sum(axis=1)).ravel().astype(np.float64))
            else:
                parts.append(np.asarray(chunk, dtype=np.float64).sum(axis=1))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)

    def _depth_sparse_sample(
        self,
        X_per_file,
        sample_global,
        offsets,
        chunk_size,
        scipy_sparse,
    ) -> np.ndarray:
        """Per-row sum on the cell sample, operating on sparse chunks
        directly. No densify, no gene permutation, no HVG slice — sums are
        column-order-invariant so all three are pure overhead for depth.
        """
        from tqdm.auto import tqdm as _tqdm

        sample_file_ids = np.searchsorted(offsets[1:], sample_global, side="right")
        sample_local = sample_global - offsets[sample_file_ids]
        plan = []
        for f_idx in range(len(X_per_file)):
            rows = sample_local[sample_file_ids == f_idx]
            if rows.size:
                phys = self._obs_int_indices_per_file[f_idx][rows]
                plan.append((f_idx, phys))
        total_chunks = sum((p.size + chunk_size - 1) // chunk_size for _, p in plan)

        parts: list[np.ndarray] = []
        pbar = _tqdm(
            total=total_chunks,
            desc=f"computing dataset stats (sampled {sample_global.size:,} cells)",
        )
        for f_idx, phys in plan:
            for start in range(0, phys.size, chunk_size):
                chunk_phys = np.sort(phys[start : start + chunk_size])
                chunk = X_per_file[f_idx][chunk_phys]
                if scipy_sparse.issparse(chunk):
                    # CSR row sum: O(nnz) without densification.
                    row_sums = np.asarray(chunk.sum(axis=1)).ravel()
                else:
                    row_sums = np.asarray(chunk).sum(axis=1)
                parts.append(row_sums.astype(np.float64))
                pbar.update(1)
        pbar.close()
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)

    def _read_dense_sample_aligned(
        self,
        X_per_file,
        sample_global,
        offsets,
        scipy_sparse,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read sample cells densified + projected to canonical gene order
        + HVG-sliced. Used for expression-value statistics where values
        must match the model's input space.

        Returns ``(sample, full_row_sums)`` where ``full_row_sums`` is the
        per-cell sum of the *full native panel* (computed before any HVG
        slicing). Callers that re-apply ``apply_normalize`` to the sample
        for stats reporting must use these full sums — sums on the
        HVG-sliced ``sample`` would inflate CP10K by ~1/(HVG share of
        library), since the HVG row-sum is a strict subset of the library.
        """
        sample_file_ids = np.searchsorted(offsets[1:], sample_global, side="right")
        sample_local = sample_global - offsets[sample_file_ids]
        chunks: list[np.ndarray] = []
        sums: list[np.ndarray] = []
        for f_idx in range(len(X_per_file)):
            mask = sample_file_ids == f_idx
            if not mask.any():
                continue
            phys = np.sort(self._obs_int_indices_per_file[f_idx][sample_local[mask]])
            chunk_raw = X_per_file[f_idx][phys]

            # Full-panel sum BEFORE any slicing (used for normalize CP10K).
            if scipy_sparse.issparse(chunk_raw):
                sums.append(
                    np.asarray(chunk_raw.sum(axis=1)).ravel().astype(np.float64)
                )
            else:
                sums.append(np.asarray(chunk_raw, dtype=np.float64).sum(axis=1))

            # Fast path: when every top-N HVG is present in this file's native
            # panel, sparse fancy-index columns to HVG order BEFORE densifying
            # — densifies only (chunk, num_hvg) instead of (chunk, n_native).
            # On Parse this drops the per-chunk dense allocation from ~660 MB
            # to ~32 MB, ~20× speedup for the expression-stats sample.
            hvg_no_missing = (
                self._hvg_no_missing_per_file is not None
                and self._hvg_adjusted_per_file is not None
                and self._hvg_no_missing_per_file[f_idx]
            )
            if hvg_no_missing and scipy_sparse.issparse(chunk_raw):
                hvg_adj = self._hvg_adjusted_per_file[f_idx]
                chunk = np.asarray(
                    chunk_raw[:, hvg_adj].toarray(),
                    dtype=np.float32,
                )
            else:
                # Slow path (target genes missing from this file's native
                # panel, or non-sparse input): full densify, permute, slice.
                chunk = chunk_raw
                if scipy_sparse.issparse(chunk):
                    chunk = chunk.toarray()
                chunk = np.asarray(chunk, dtype=np.float32)
                if self._gene_perms_per_file is not None:
                    chunk = self._apply_permutation_dense(
                        chunk,
                        self._gene_perms_per_file[f_idx],
                        self._n_native_per_file[f_idx],
                    )
                if self._num_hvg is not None:
                    chunk = chunk[:, : self._num_hvg]
            chunks.append(chunk)
        if not chunks:
            return (
                np.zeros((0, self._n_vars), dtype=np.float32),
                np.zeros(0, dtype=np.float64),
            )
        return np.concatenate(chunks, axis=0), np.concatenate(sums)

    def _make_data_stats(
        self,
        n_cells: int,
        depth: np.ndarray,
        log_depth: np.ndarray,
        flat: np.ndarray,
    ) -> DataStats:
        median_lib = (
            self._median_library_size
            if self._median_library_size is not None
            else float(np.median(depth)) if depth.size else 0.0
        )
        # count_max: prefer the resolved dataset value (the property that sizes
        # embeddings); fall back to the sample max for log1p data where it's None.
        count_max = (
            float(self._count_max)
            if self._count_max is not None
            else float(flat.max()) if flat.size else 0.0
        )
        return DataStats(
            num_cells=n_cells,
            num_genes=self._n_vars,
            num_perts=self._n_perts,
            log_lib_mean=float(log_depth.mean()),
            log_lib_std=float(log_depth.std()),
            median_library_size=float(median_lib),
            count_max=count_max,
            depth_mean=float(depth.mean()) if depth.size else 0.0,
            depth_median=float(np.median(depth)) if depth.size else 0.0,
            nonzero_frac=(
                float(np.count_nonzero(flat) / flat.size) if flat.size else 0.0
            ),
            count_p99=float(np.percentile(flat, 99)) if flat.size else 0.0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pert_emb(self, path: str) -> torch.Tensor:
        raw = torch.load(path, weights_only=True)

        if isinstance(raw, torch.Tensor):
            if raw.shape[0] != self.num_perts:
                raise ValueError(
                    f"pert_mapping tensor has {raw.shape[0]} rows but dataset "
                    f"has {self.num_perts} unique perturbations."
                )
            return raw.float()

        if isinstance(raw, dict):
            sample_emb = next(iter(raw.values()))
            emb_dim = sample_emb.shape[-1]
            matrix = torch.zeros(self.num_perts, emb_dim, dtype=torch.float32)
            for name, emb in raw.items():
                if name in self.pert_to_idx:
                    matrix[self.pert_to_idx[name]] = emb.float()
            missing = [p for p in self.unique_perts if p not in raw]
            if missing:
                self._log_warning(
                    f"{len(missing)} dataset perturbation(s) have no entry in "
                    f"pert_mapping and will use zero embeddings "
                    f"(e.g. {missing[:3]})."
                )
            return matrix

        raise TypeError(
            f"pert_mapping_path must contain a Tensor or dict[str, Tensor], "
            f"got {type(raw).__name__}."
        )

    def _log_info(self, message: str) -> None:
        try:
            self._logger.info(message, main_process_only=True, stacklevel=2)
        except TypeError:
            try:
                self._logger.info(message, stacklevel=2)
            except TypeError:
                self._logger.info(message)

    def _log_warning(self, message: str) -> None:
        try:
            self._logger.warning(message, main_process_only=True, stacklevel=2)
        except TypeError:
            try:
                self._logger.warning(message, stacklevel=2)
            except TypeError:
                self._logger.warning(message)


# ---------------------------------------------------------------------------
# Train / val split helper
# ---------------------------------------------------------------------------


def _make_dataset_view(
    ds: PerturbationDataset, indices: np.ndarray
) -> PerturbationDataset:
    """
    Return a lightweight Dataset shell that shares all heavy arrays with *ds*
    by reference (no data is copied) but exposes only the given cell indices.
    """
    view = object.__new__(PerturbationDataset)
    # Eager-mode arrays (None in lazy mode)
    view.counts = ds.counts
    view._count_max = ds.count_max
    # Shared unified indices
    view._pert_indices = ds._pert_indices
    view._cell_type_indices = ds._cell_type_indices
    view._donor_indices = ds._donor_indices
    view._donor_col = ds._donor_col
    view.pert_emb = ds.pert_emb
    view.unique_perts = ds.unique_perts
    view.pert_to_idx = ds.pert_to_idx
    view.unique_cell_types = ds.unique_cell_types
    view.cell_type_to_idx = ds.cell_type_to_idx
    view.unique_donors = ds.unique_donors
    view.donor_to_idx = ds.donor_to_idx
    view._n_vars = ds.num_genes
    view._n_perts = ds.num_perts
    view._n_cell_types = ds._n_cell_types
    view._n_obs = ds._n_obs
    view._logger = ds._logger
    view._indices = indices
    # Lazy / multi-file fields
    view._load_lazily = ds._load_lazily
    view._h5ad_paths = ds._h5ad_paths
    view._h5ad_path = ds._h5ad_path
    view._file_meta = ds._file_meta
    view._file_row_offsets = ds._file_row_offsets
    view._apply_normalize = ds._apply_normalize
    view._config_target_sum = ds._config_target_sum
    view._per_file_target_sum = ds._per_file_target_sum
    view._median_library_size = ds._median_library_size
    view._apply_log1p = ds._apply_log1p
    view._use_binning = ds._use_binning
    view._obs_int_indices_per_file = ds._obs_int_indices_per_file
    view._obs_int_indices = ds._obs_int_indices
    view._lazy_X_per_file = None  # MUST be None: each worker opens its own
    view._lazy_h5_files = None
    view._lazy_X = None
    # Gene-order permutation state
    view._target_genes = ds._target_genes
    view._num_hvg = ds._num_hvg
    view._gene_perms_per_file = ds._gene_perms_per_file
    view._n_native_per_file = ds._n_native_per_file
    view._hvg_adjusted_per_file = ds._hvg_adjusted_per_file
    view._hvg_no_missing_per_file = ds._hvg_no_missing_per_file
    view._library_size = ds._library_size
    # derive_log1p_via_lib_full state (lazy mode only; eager mode keeps None)
    view._derive_log1p_per_file = getattr(ds, "_derive_log1p_per_file", None)
    view._lib_full_per_file = getattr(ds, "_lib_full_per_file", None)
    view._median_lib_per_file = getattr(ds, "_median_lib_per_file", None)
    return view


def split_dataset(
    ds: PerturbationDataset,
    val_frac: float = 0.1,
    val_pert_frac: float = 0.0,
    seed: int = 42,
    ctrl_label: str = "ctrl",
    ctrl_labels: list[str] | None = None,
) -> tuple[PerturbationDataset, PerturbationDataset]:
    """
    Split an existing PerturbationDataset into train and val subsets.

    Both returned datasets share the **same** underlying data by reference —
    no data is copied.

    Parameters
    ----------
    ds:
        A fully constructed PerturbationDataset.
    val_frac:
        Fraction of cells for validation (cell-level split, used when
        ``val_pert_frac == 0``).
    val_pert_frac:
        Fraction of non-control perturbation identities to hold out.
        When > 0 overrides *val_frac*.
    seed:
        RNG seed for reproducibility.
    ctrl_label:
        Single perturbation label for unperturbed control cells.
    ctrl_labels:
        Multiple control labels (e.g. ``["SAFE_TARGET", "NO-TARGET"]``).
        When provided, overrides *ctrl_label*.
    """
    if val_pert_frac > 0.0:
        return split_dataset_by_pert(
            ds,
            val_pert_frac=val_pert_frac,
            seed=seed,
            ctrl_label=ctrl_label,
            ctrl_labels=ctrl_labels,
        )

    n = len(ds)
    all_idx = np.arange(n, dtype=np.int64)
    if ds._indices is not None:
        all_idx = ds._indices.copy()

    rng = np.random.default_rng(seed)
    rng.shuffle(all_idx)
    split = int(n * (1.0 - val_frac))

    return _make_dataset_view(ds, all_idx[:split]), _make_dataset_view(
        ds, all_idx[split:]
    )


def split_dataset_by_pert(
    ds: PerturbationDataset,
    val_pert_frac: float = 0.1,
    seed: int = 42,
    ctrl_label: str = "ctrl",
    ctrl_labels: list[str] | None = None,
) -> tuple[PerturbationDataset, PerturbationDataset]:
    """
    Split by holding out a fraction of perturbation identities from training.

    Control cells appear in **both** train and val (split proportionally by
    ``val_pert_frac``). All heavy arrays are shared by reference.

    Parameters
    ----------
    ds:
        A fully constructed PerturbationDataset.
    val_pert_frac:
        Fraction of non-control perturbation identities to hold out.
    seed:
        RNG seed for reproducibility.
    ctrl_label:
        Single control perturbation label.
    ctrl_labels:
        Multiple control labels (e.g. ``["SAFE_TARGET", "NO-TARGET"]``).
        When non-empty, overrides *ctrl_label*.
    """
    rng = np.random.default_rng(seed)

    # Resolve the effective set of control perturbation indices
    effective_ctrl_labels: list[str] = ctrl_labels if ctrl_labels else [ctrl_label]
    ctrl_idxs: frozenset[int] = frozenset(
        ds.pert_to_idx[lbl] for lbl in effective_ctrl_labels if lbl in ds.pert_to_idx
    )

    non_ctrl_perts = np.array(
        [i for i in range(ds.num_perts) if i not in ctrl_idxs],
        dtype=np.int64,
    )

    n_val_perts = max(1, round(len(non_ctrl_perts) * val_pert_frac))
    shuffled = rng.permutation(non_ctrl_perts)
    val_pert_set = set(shuffled[:n_val_perts].tolist())
    train_pert_set = set(shuffled[n_val_perts:].tolist())

    all_idx = (
        ds._indices if ds._indices is not None else np.arange(len(ds), dtype=np.int64)
    )
    cell_perts = ds._pert_indices[all_idx]

    train_mask = np.isin(cell_perts, list(train_pert_set))
    val_mask = np.isin(cell_perts, list(val_pert_set))

    if ctrl_idxs:
        ctrl_mask = np.isin(cell_perts, list(ctrl_idxs))
        ctrl_cells = all_idx[ctrl_mask]
        rng.shuffle(ctrl_cells)
        n_val_ctrl = max(1, round(len(ctrl_cells) * val_pert_frac))
        train_idx = np.concatenate([all_idx[train_mask], ctrl_cells[n_val_ctrl:]])
        val_idx = np.concatenate([all_idx[val_mask], ctrl_cells[:n_val_ctrl]])
        ctrl_info = (
            f" ({n_val_ctrl} / {len(ctrl_cells)} ctrl cells in val"
            f" across {len(ctrl_idxs)} ctrl label(s))"
        )
    else:
        train_idx = all_idx[train_mask]
        val_idx = all_idx[val_mask]
        ctrl_info = ""

    ds._log_info(
        f"Pert split: {len(train_pert_set)} train perts, {len(val_pert_set)} held-out val perts | "
        f"{len(train_idx):,} train cells, {len(val_idx):,} val cells{ctrl_info}"
    )

    return _make_dataset_view(ds, train_idx), _make_dataset_view(ds, val_idx)


def split_dataset_by_donor_pert(
    ds: PerturbationDataset,
    test_donors: list[str] | tuple[str, ...],
    test_perts: list[str] | tuple[str, ...],
) -> tuple[PerturbationDataset, PerturbationDataset]:
    """
    Donor-holdout split: test = (cell.donor ∈ test_donors) ∧ (cell.pert ∈ test_perts).
    Train is the complement — every cell from a non-held-out donor, plus every
    cell from a held-out donor whose perturbation is *not* in ``test_perts``
    (controls and any non-test perturbations leak back to training).

    Generic donor × perturbation holdout. The dataset must have been built
    with ``DatasetConfig.donor_col`` set so ``_donor_indices`` is populated.

    Both returned datasets share the same underlying arrays by reference.
    """
    if ds._donor_indices is None:
        raise ValueError(
            "split_dataset_by_donor_pert requires the dataset to be built with "
            "DatasetConfig.donor_col set (e.g. donor_col='donor')."
        )

    test_donor_idxs = np.array(
        [ds.donor_to_idx[d] for d in test_donors if d in ds.donor_to_idx],
        dtype=np.int64,
    )
    missing_donors = [d for d in test_donors if d not in ds.donor_to_idx]
    if missing_donors:
        ds._log_warning(
            f"split_dataset_by_donor_pert: {len(missing_donors)} test donor(s) "
            f"not found in dataset and will be ignored: {missing_donors}"
        )
    if test_donor_idxs.size == 0:
        raise ValueError(
            "split_dataset_by_donor_pert: no requested test donors are present "
            f"in the dataset. Available donors: {ds.unique_donors}"
        )

    test_pert_idxs = np.array(
        [ds.pert_to_idx[p] for p in test_perts if p in ds.pert_to_idx],
        dtype=np.int64,
    )
    missing_perts = [p for p in test_perts if p not in ds.pert_to_idx]
    if missing_perts:
        ds._log_warning(
            f"split_dataset_by_donor_pert: {len(missing_perts)} test perturbation(s) "
            f"not found in dataset and will be ignored: {missing_perts}"
        )
    if test_pert_idxs.size == 0:
        raise ValueError(
            "split_dataset_by_donor_pert: no requested test perturbations are "
            "present in the dataset."
        )

    all_idx = (
        ds._indices if ds._indices is not None else np.arange(len(ds), dtype=np.int64)
    )
    cell_donors = ds._donor_indices[all_idx]
    cell_perts = ds._pert_indices[all_idx]

    test_mask = np.isin(cell_donors, test_donor_idxs) & np.isin(
        cell_perts, test_pert_idxs
    )
    train_idx = all_idx[~test_mask]
    test_idx = all_idx[test_mask]

    ds._log_info(
        f"Donor-pert split: {len(test_donor_idxs)} test donor(s) × "
        f"{len(test_pert_idxs)} test pert(s) → "
        f"{len(train_idx):,} train cells, {len(test_idx):,} test cells"
    )

    return _make_dataset_view(ds, train_idx), _make_dataset_view(ds, test_idx)


def split_dataset_by_celltype_pert(
    ds: PerturbationDataset,
    test_cell_types: list[str] | tuple[str, ...],
    test_perts: list[str] | tuple[str, ...],
) -> tuple[PerturbationDataset, PerturbationDataset]:
    """
    Cell-type-holdout split: test = (cell.cell_type ∈ test_cell_types) ∧
    (cell.pert ∈ test_perts). Train is the complement — every cell from a
    non-held-out cell type, plus every cell from a held-out cell type whose
    perturbation is *not* in ``test_perts`` (controls and any non-test
    perturbations leak back to training).

    Implements the scLDM (Palla et al. 2025) HepG2-holdout split for
    Replogle-Nadig when fed the ``cell_type`` and ``pert`` lists from
    ``assets/scldm_replogle_test.json`` (loaded via
    ``train_utils.load_split_spec`` and dispatched by
    ``train_utils.resolve_train_val_split``).

    Both returned datasets share the same underlying arrays by reference.
    """
    test_ct_idxs = np.array(
        [ds.cell_type_to_idx[c] for c in test_cell_types if c in ds.cell_type_to_idx],
        dtype=np.int64,
    )
    missing_cts = [c for c in test_cell_types if c not in ds.cell_type_to_idx]
    if missing_cts:
        ds._log_warning(
            f"split_dataset_by_celltype_pert: {len(missing_cts)} test cell type(s) "
            f"not found in dataset and will be ignored: {missing_cts}"
        )
    if test_ct_idxs.size == 0:
        raise ValueError(
            "split_dataset_by_celltype_pert: no requested test cell types are "
            f"present in the dataset. Available cell types: {ds.unique_cell_types}"
        )

    test_pert_idxs = np.array(
        [ds.pert_to_idx[p] for p in test_perts if p in ds.pert_to_idx],
        dtype=np.int64,
    )
    missing_perts = [p for p in test_perts if p not in ds.pert_to_idx]
    if missing_perts:
        ds._log_warning(
            f"split_dataset_by_celltype_pert: {len(missing_perts)} test perturbation(s) "
            f"not found in dataset and will be ignored: {missing_perts}"
        )
    if test_pert_idxs.size == 0:
        raise ValueError(
            "split_dataset_by_celltype_pert: no requested test perturbations are "
            "present in the dataset."
        )

    all_idx = (
        ds._indices if ds._indices is not None else np.arange(len(ds), dtype=np.int64)
    )
    cell_cts = ds._cell_type_indices[all_idx]
    cell_perts = ds._pert_indices[all_idx]

    test_mask = np.isin(cell_cts, test_ct_idxs) & np.isin(cell_perts, test_pert_idxs)
    train_idx = all_idx[~test_mask]
    test_idx = all_idx[test_mask]

    ds._log_info(
        f"Cell-type-pert split: {len(test_ct_idxs)} test cell type(s) × "
        f"{len(test_pert_idxs)} test pert(s) → "
        f"{len(train_idx):,} train cells, {len(test_idx):,} test cells"
    )

    return _make_dataset_view(ds, train_idx), _make_dataset_view(ds, test_idx)


def split_dataset_by_celltype_donor_pert(
    ds: PerturbationDataset,
    test_cell_types: list[str] | tuple[str, ...],
    test_donors: list[str] | tuple[str, ...],
    test_perts: list[str] | tuple[str, ...],
) -> tuple[PerturbationDataset, PerturbationDataset]:
    """
    3-way intersection holdout: test = (cell.cell_type ∈ test_cell_types) ∧
    (cell.donor ∈ test_donors) ∧ (cell.pert ∈ test_perts). Train is the
    complement — every other cell, including cells from the held-out donors
    that don't satisfy the cell-type constraint, cells of the held-out cell
    type from non-held-out donors (cell-type leakage allowed), and cells of
    the held-out cell type / donor whose pert is not in ``test_perts``.

    Generic cell-type × donor × perturbation holdout. The dataset must have
    been built with ``DatasetConfig.donor_col`` set so ``_donor_indices`` is
    populated.

    Both returned datasets share the same underlying arrays by reference.
    """
    if ds._donor_indices is None:
        raise ValueError(
            "split_dataset_by_celltype_donor_pert requires the dataset to be "
            "built with DatasetConfig.donor_col set (e.g. donor_col='donor')."
        )

    test_ct_idxs = np.array(
        [ds.cell_type_to_idx[c] for c in test_cell_types if c in ds.cell_type_to_idx],
        dtype=np.int64,
    )
    missing_cts = [c for c in test_cell_types if c not in ds.cell_type_to_idx]
    if missing_cts:
        ds._log_warning(
            f"split_dataset_by_celltype_donor_pert: {len(missing_cts)} test "
            f"cell type(s) not found in dataset and will be ignored: {missing_cts}"
        )
    if test_ct_idxs.size == 0:
        raise ValueError(
            "split_dataset_by_celltype_donor_pert: no requested test cell types "
            f"are present in the dataset. Available cell types: {ds.unique_cell_types}"
        )

    test_donor_idxs = np.array(
        [ds.donor_to_idx[d] for d in test_donors if d in ds.donor_to_idx],
        dtype=np.int64,
    )
    missing_donors = [d for d in test_donors if d not in ds.donor_to_idx]
    if missing_donors:
        ds._log_warning(
            f"split_dataset_by_celltype_donor_pert: {len(missing_donors)} test "
            f"donor(s) not found in dataset and will be ignored: {missing_donors}"
        )
    if test_donor_idxs.size == 0:
        raise ValueError(
            "split_dataset_by_celltype_donor_pert: no requested test donors are "
            f"present in the dataset. Available donors: {ds.unique_donors}"
        )

    test_pert_idxs = np.array(
        [ds.pert_to_idx[p] for p in test_perts if p in ds.pert_to_idx],
        dtype=np.int64,
    )
    missing_perts = [p for p in test_perts if p not in ds.pert_to_idx]
    if missing_perts:
        ds._log_warning(
            f"split_dataset_by_celltype_donor_pert: {len(missing_perts)} test "
            f"perturbation(s) not found in dataset and will be ignored: {missing_perts}"
        )
    if test_pert_idxs.size == 0:
        raise ValueError(
            "split_dataset_by_celltype_donor_pert: no requested test perturbations "
            "are present in the dataset."
        )

    all_idx = (
        ds._indices if ds._indices is not None else np.arange(len(ds), dtype=np.int64)
    )
    cell_cts = ds._cell_type_indices[all_idx]
    cell_donors = ds._donor_indices[all_idx]
    cell_perts = ds._pert_indices[all_idx]

    test_mask = (
        np.isin(cell_cts, test_ct_idxs)
        & np.isin(cell_donors, test_donor_idxs)
        & np.isin(cell_perts, test_pert_idxs)
    )
    train_idx = all_idx[~test_mask]
    test_idx = all_idx[test_mask]

    ds._log_info(
        f"Cell-type-donor-pert split: {len(test_ct_idxs)} test cell type(s) × "
        f"{len(test_donor_idxs)} test donor(s) × {len(test_pert_idxs)} test pert(s) "
        f"→ {len(train_idx):,} train cells, {len(test_idx):,} test cells"
    )

    return _make_dataset_view(ds, train_idx), _make_dataset_view(ds, test_idx)


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------


def make_dataloader_from_dataset(
    ds: PerturbationDataset,
    batch_size: int = 256,
    num_workers: int = 8,
    shuffle: bool = True,
    **dataloader_kwargs: Any,
) -> DataLoader:
    """
    Wrap a PerturbationDataset in a DataLoader tuned for fast training.

    - ``pin_memory=True``         faster host→GPU transfer
    - ``persistent_workers=True`` workers survive between epochs; no per-epoch
                                   re-init cost and no file reopening (important
                                   for lazy-loaded datasets)
    """
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        **dataloader_kwargs,
    )


def make_dataloader(
    h5ad_path: str,
    pert_col: str = "gene",
    pert_mapping_path: str | None = None,
    batch_size: int = 256,
    num_workers: int = 8,
    shuffle: bool = True,
    indices: np.ndarray | None = None,
    **dataloader_kwargs: Any,
) -> DataLoader:
    """
    Convenience: build a PerturbationDataset and wrap it in a DataLoader.

    For train/val splits, prefer constructing the dataset once and using
    ``split_dataset`` + ``make_dataloader_from_dataset`` to avoid loading
    the count matrix twice.
    """
    dataset_config = DatasetConfig(
        h5ad_path=h5ad_path,
        pert_col=pert_col,
        pert_mapping_path=pert_mapping_path,
    )
    ds = PerturbationDataset(config=dataset_config, indices=indices)
    return make_dataloader_from_dataset(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        **dataloader_kwargs,
    )


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test the perturbation dataloader."
    )
    parser.add_argument("--h5ad", required=True, help="Path to .h5ad file")
    parser.add_argument("--pert_col", default="gene")
    parser.add_argument(
        "--pert_mapping", default=None, help="Optional .pt pert embedding file"
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_batches", type=int, default=10, help="Batches to time")
    parser.add_argument("--lazy", action="store_true", help="Use lazy loading")
    args = parser.parse_args()

    dataset_config = DatasetConfig(
        h5ad_path=args.h5ad,
        pert_col=args.pert_col,
        pert_mapping_path=args.pert_mapping,
        load_lazily=args.lazy,
    )
    ds = PerturbationDataset(config=dataset_config)

    print(
        f"\nDataset: {ds.num_cells:,} cells, {ds.num_genes} genes, {ds.num_perts} unique perts"
    )
    print(f"Mode:    {'lazy (HDF5-backed)' if args.lazy else 'eager (in-memory)'}")

    train_ds, val_ds = split_dataset(ds, val_frac=0.1)
    print(f"Split  : {len(train_ds):,} train / {len(val_ds):,} val")

    dl = make_dataloader_from_dataset(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(
        f"\nTiming first {args.n_batches} batches  "
        f"(batch_size={args.batch_size}, num_workers={args.num_workers}) …"
    )

    t0 = time.perf_counter()
    for i, batch in enumerate(dl):
        if i == 0:
            print("  Keys     :", list(batch.keys()))
            print("  counts   :", batch["counts"].shape, batch["counts"].dtype)
            print("  pert_idx :", batch["pert_idx"].shape, batch["pert_idx"].dtype)
            if "pert_emb" in batch:
                print("  pert_emb :", batch["pert_emb"].shape, batch["pert_emb"].dtype)
        if i + 1 >= args.n_batches:
            break

    elapsed = time.perf_counter() - t0
    cells = args.n_batches * args.batch_size
    print(
        f"\n  {args.n_batches} batches in {elapsed:.2f}s  →  "
        f"{cells / elapsed:,.0f} cells/sec"
    )
