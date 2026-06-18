from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    h5ad_path: str = ""  # single-file path; set via YAML or CLI
    # Multi-file loading. When non-empty, overrides h5ad_path. All files must
    # have the same gene set (n_vars + var_names order). Per-file metadata
    # (cell type, pert column, control label) is looked up by basename in
    # sc_evae.training.constants.DATASET_METADATA.
    h5ad_paths: list[str] = field(default_factory=list)
    pert_col: str = "gene"
    # Optional .obs column to load alongside cell_type and the perturbation
    # column. When set, the loader builds a donor vocabulary on the dataset
    # (``unique_donors`` / ``donor_to_idx`` / ``_donor_indices``) so train/test
    # splits can key on it (see ``split_dataset_by_donor_pert``). The column
    # must exist on every loaded file. ``None`` disables.
    donor_col: str | None = None
    ctrl_label: str = "non-targeting"  # label for unperturbed control cells in pert_col
    # Multiple control labels (e.g. ["SAFE_TARGET", "NO-TARGET"] for Pfizer).
    # When non-empty, overrides ctrl_label in train/val split logic.
    ctrl_labels: list[str] = field(default_factory=list)
    pert_mapping_path: str | None = None
    min_genes: int = 200
    min_cells: int = 3
    # Normalization target for sc.pp.normalize_total. ``None`` (default) uses
    # the per-file median library size — i.e. each file's cells are scaled to
    # that file's median total count (scanpy's normalize_total(target_sum=None)
    # semantics, and what the released datasets' offline log1p layers used).
    # Set a float (e.g. 1e4) to force a fixed CP-N target across all files.
    target_sum: float | None = None
    apply_normalize: bool = True  # apply sc.pp.normalize_total(target_sum)
    apply_log1p: bool = True  # apply sc.pp.log1p after normalization
    batch_size: int = 512
    num_workers: int = 8
    # Cell-level random split (used when val_pert_frac == 0.0).
    val_frac: float = 0.0
    # Perturbation-level holdout split. When > 0, this fraction of non-control
    # perturbation identities are excluded from training entirely and used as
    # the val set (control cells appear in both). Overrides val_frac when set.
    val_pert_frac: float = 0.1
    # When True, X is never loaded into RAM. Rows are read from the HDF5 file
    # lazily in __getitem__, one at a time, with preprocessing applied per-row.
    # Required for datasets too large to fit in RAM (e.g. Pfizer ~40 GB).
    # filter_genes is skipped (requires a full X pass); filter_cells uses the
    # obs['n_genes'] column if present, otherwise is also skipped.
    load_lazily: bool = False
    # Binning mode applied after the existing normalize/log1p chain.
    #   "none"     — no binning (default)
    #   "quantile" — 20-class hardcoded integer-range bins (see COUNT_BIN_UPPER
    #                in sc_evae.training.tokenizer).  Forces count_max =
    #                19.  Requires apply_log1p=False (operates on integer counts).
    use_binning: str = "none"
    # Path to a JSON file describing a cell-type × perturbation holdout. When
    # set, takes precedence over val_frac / val_pert_frac and routes through a
    # cell-type-pert holdout (``split_dataset_by_celltype_pert``) instead of the
    # random / pert-frac splits.
    #   None — fallback to val_frac / val_pert_frac.
    #   path — a JSON file ``{"cell_type": [...], "pert": [...]}`` whose
    #          ``cell_type`` × ``pert`` cross product is held out for testing.
    # The released scLDM holdouts live in ``assets/``:
    #   assets/scldm_replogle_test.json  — Replogle-Nadig HepG2 × 372 perts.
    #   assets/scldm_parse1m_test.json   — Parse-PBMC CD4 Naive × 27 cytokines.
    # To run a different holdout, point this at any JSON of the same shape — no
    # code change needed. See ``train_utils.resolve_train_val_split`` for the
    # dispatcher and ``train_utils.load_split_spec`` for the loader/validator.
    split_path: str | None = None
    # Path to a newline-separated canonical gene-name file. When set, the loader
    # permutes each h5ad's columns to match this order (zero-filling any genes
    # missing from the source file). The dataset's effective num_genes becomes
    # len(gene_order). Per-file native column orders may differ — each file gets
    # its own permutation. Permutation happens AFTER normalize_total + log1p, so
    # depth normalization always uses the file's native (full) gene panel.
    gene_order_path: str | None = None
    # When set, slice the permuted columns to the first ``num_hvg`` entries
    # (i.e. the top-N HVGs assuming gene_order_path is HVG-rank-ordered). The
    # dataset's effective num_genes becomes ``num_hvg``. Requires gene_order_path.
    num_hvg: int | None = None
    # Per-column inclusion filter on .obs values, applied at dataset-init time.
    # Example:  {condition: [Untreated]}  →  keep only cells where obs.condition
    # is in [Untreated]. Multiple column entries are AND-ed.
    #
    # YAML values REPLACE per-file defaults from DATASET_METADATA at the column
    # level. An EMPTY list for a column means "no filter on this column"
    # (clears any per-file default). Columns mentioned only here (not in any
    # per-file default) are applied leniently — files lacking the column log a
    # warning and skip the filter rather than dropping all cells. Columns
    # declared in per-file metadata are strict — files missing the column
    # raise ValueError (catches typos in DATASET_METADATA).
    obs_filters: dict[str, list[str]] = field(default_factory=dict)
    # Optional path to a precomputed DEG table (.pt produced by
    # ``scripts/utils/precompute_deg_table.py`` or saved by a prior training
    # run as ``{run_dir}/deg_table.pt``). When set, training skips the
    # expensive build_deg_table_from_dataset pass and loads from disk; a
    # shape check (mask width == num_genes, indices length == top_k) catches
    # config drift. When None, falls back to
    # ``constants.resolve_default_deg_table_path(dataset_cfg)`` (basename or
    # frozenset lookup), and finally to recomputing on the fly.
    deg_table_path: str | None = None

    def __post_init__(self) -> None:
        if self.num_hvg is not None:
            if self.gene_order_path is None:
                raise ValueError(
                    "DatasetConfig.num_hvg requires gene_order_path to be set: "
                    "without a canonical gene order, 'top-N' is undefined."
                )
            if self.num_hvg <= 0:
                raise ValueError(
                    f"DatasetConfig.num_hvg must be positive, got {self.num_hvg}"
                )

        if self.split_path is not None and not self.split_path.endswith(".json"):
            raise ValueError(
                f"DatasetConfig.split_path={self.split_path!r} must be a path to a "
                f'.json file of shape {{"cell_type": [...], "pert": [...]}}.'
            )
