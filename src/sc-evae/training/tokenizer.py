"""
tokenizer.py
------------
Count tokenizer: maps raw integer gene counts to discrete "quantile" bin tokens
and back to representative continuous values.

This is the discretisation that turns a cell's raw count vector into integer
token IDs for the cross-entropy / discrete-latent heads (``use_binning:
quantile`` in ``DatasetConfig``). ``none`` is the no-op pass-through mode kept so
``PerturbationDataset`` can stay binning-agnostic.

Public API (consumed by ``data_loader.py`` and ``train_utils.py``):
  * ``_apply_binning(arr, mode)``      — counts → bin IDs (no-op for "none")
  * ``_binning_count_max(mode)``       — fixed max bin index for a mode
  * ``bin_id_to_value(bin_ids, mode)`` — bin IDs → representative counts
  * ``_BINNING_MODES``                 — the set of valid modes
"""

from __future__ import annotations

import numpy as np
import torch

# --- "quantile" mode: 20-class hardcoded integer-range bins -----------------
# Bin i covers (COUNT_BIN_UPPER[i-1], COUNT_BIN_UPPER[i]], with bin 0 = {0}
# and bin 19 (the lump bucket) = {256, 257, ...}.  Boundaries chosen from the
# count distribution on Replogle K562 (raw counts) and validated against
# Parse 10M cytokine: bins 0..14 cover counts 0..63 with progressively coarser
# resolution (matches the heavy-tail decay of single-cell counts), and bins
# 15..19 cover the high-expression tail. The high-tail boundaries (95, 127,
# 191, 255) are chosen so each bounded bin holds 0.5–4M (cell, gene) entries
# on Replogle (within an order of magnitude of bin 14), and so 99.69% of
# top-DEG per-pert-mean entries on Replogle and 100.00% on Parse cytokine
# land in bounded bins rather than the open lump bucket.
COUNT_BIN_UPPER = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 15, 23, 31, 47, 63, 95, 127, 191, 255],
    dtype=np.int64,
)
COUNT_BIN_NUM_CLASSES = len(COUNT_BIN_UPPER) + 1  # = 20
COUNT_BIN_MAX_INDEX = COUNT_BIN_NUM_CLASSES - 1  # = 19


def _bin_counts_integer(counts: np.ndarray) -> np.ndarray:
    """Map integer counts → bin index in [0, COUNT_BIN_MAX_INDEX].

    Processes in row chunks when the input is a large 2-D matrix: the naive
    ``counts.astype(np.int64)`` materialises a full int64 copy (8 B/cell)
    plus another int64 searchsorted output, which on a 1.27M × 2000 dense
    matrix is ~38 GB transient and OOMs.
    """
    if counts.ndim == 2 and counts.size > 1_000_000:
        out = np.empty(counts.shape, dtype=np.float32)
        chunk = 50_000
        for i in range(0, counts.shape[0], chunk):
            j = min(i + chunk, counts.shape[0])
            idx = np.searchsorted(
                COUNT_BIN_UPPER, counts[i:j].astype(np.int64), side="left"
            )
            out[i:j] = np.minimum(idx, COUNT_BIN_MAX_INDEX).astype(np.float32)
        return out
    idx = np.searchsorted(COUNT_BIN_UPPER, counts.astype(np.int64), side="left")
    return np.minimum(idx, COUNT_BIN_MAX_INDEX).astype(np.float32)


_BINNING_MODES = ("none", "quantile")


def _apply_binning(arr: np.ndarray, mode: str) -> np.ndarray:
    """Dispatch to the binning helper for the given mode (no-op for 'none')."""
    if mode == "quantile":
        return _bin_counts_integer(arr)
    return arr


def _binning_count_max(mode: str) -> int | None:
    """Return the fixed count_max for a binning mode, or None for 'none'."""
    if mode == "quantile":
        return COUNT_BIN_MAX_INDEX
    return None


# --- Bin → representative-value lookups (used for metric-space conversion) ---
# QUANTILE_BIN_VALUES[i] is the representative integer count for bin i in
# "quantile" mode. Bins 0..7 are singletons, so the value equals the index.
# Bins 8..14 cover small ranges → arithmetic midpoint. Bins 15..19 cover the
# high-expression tail; values are *empirical conditional means* of each bin
# range computed on Replogle K562 (heavy-tail-aware — preferred over the
# arithmetic midpoint because the within-bin count distribution clusters at
# the low end). Bin 19 is the open lump bucket {256, 257, ...}; the value
# 360.8 is E[count | count >= 256] on Replogle K562.
QUANTILE_BIN_VALUES = np.array(
    [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.5,
        10.5,
        13.5,
        19.5,
        27.5,
        39.5,
        55.5,
        76.5,
        109.3,
        154.3,
        219.9,
        360.8,
    ],
    dtype=np.float32,
)
assert len(QUANTILE_BIN_VALUES) == COUNT_BIN_NUM_CLASSES


def bin_id_to_value(bin_ids, mode: str):
    """
    Convert bin indices to their representative continuous values.

    Accepts a torch.Tensor or np.ndarray of bin indices (any dtype that can be
    cast to int64) and returns the same container type filled with float32 values:
      - mode="quantile" → integer-count midpoint of each bin's range.
      - mode="none"     → the input is returned unchanged.

    Out-of-range indices are clipped to the nearest valid bin.
    """
    if mode == "none":
        return bin_ids
    if mode == "quantile":
        table_np = QUANTILE_BIN_VALUES
        max_idx = COUNT_BIN_MAX_INDEX
    else:
        raise ValueError(f"Unknown binning mode: {mode!r}")

    if isinstance(bin_ids, torch.Tensor):
        idx = bin_ids.long().clamp(0, max_idx)
        table = torch.as_tensor(table_np, dtype=torch.float32, device=idx.device)
        return table[idx]
    arr = np.clip(np.asarray(bin_ids, dtype=np.int64), 0, max_idx)
    return table_np[arr]
