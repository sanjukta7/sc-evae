"""
src/sc_evae/metrics/pfizer_ranking.py
-------------------------------------
Embedding-quality metrics for the Pfizer IMRU phenotypic reversion eval.

All functions operate on numpy arrays and are independent of the sc_evae model.
Core logic ported from pfizer-opensource/phenotype_reversion/library.py
(rank_perturbations_by_similarity, get_true_and_random_clustering_scores).
"""

from __future__ import annotations

import random

import numpy as np
from scipy.spatial.distance import cosine as _cosine_distance
from scipy.stats import spearmanr
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import normalize as l2_normalize

# ---------------------------------------------------------------------------
# Metric 1: Phenotypic reversion ranking
# ---------------------------------------------------------------------------


def centroid_cosine_ranking(
    embeddings: np.ndarray,
    group_labels: list[str],
    candidate_prefix: str,
    anchor_label: str,
) -> list[tuple[str, float]]:
    """Rank perturbation groups by cosine distance of their centroid to an anchor.

    Parameters
    ----------
    embeddings:
        (N, D) float32 array of cell embeddings (need not be L2-normalized;
        normalization is applied internally per centroid).
    group_labels:
        Length-N list of strings formatted as ``"{condition}_{gene_target}"``
        for each cell.
    candidate_prefix:
        Only groups whose label starts with this string are ranked
        (e.g. ``"Treated_"``).
    anchor_label:
        The group label used as the healthy reference centroid
        (e.g. ``"Untreated_SAFE_TARGET"``).

    Returns
    -------
    List of ``(gene_target, cosine_distance)`` tuples sorted ascending
    (lowest distance = most similar to anchor = best reversion candidate).
    Ported from ``rank_perturbations_by_similarity()`` in pfizer/library.py.
    """
    labels_arr = np.array(group_labels)
    unique_labels = list(set(group_labels))

    # Compute mean centroid per group
    centroids: dict[str, np.ndarray] = {}
    for label in unique_labels:
        mask = labels_arr == label
        centroids[label] = embeddings[mask].mean(axis=0)

    if anchor_label not in centroids:
        raise ValueError(
            f"Anchor label {anchor_label!r} not found in group_labels. "
            f"Available: {sorted(unique_labels)[:10]}..."
        )

    anchor_centroid = centroids[anchor_label]

    results: list[tuple[str, float]] = []
    for label, centroid in centroids.items():
        if not label.startswith(candidate_prefix):
            continue
        gene_target = label[len(candidate_prefix) :]
        dist = _cosine_distance(centroid, anchor_centroid)
        results.append((gene_target, float(dist)))

    results.sort(key=lambda x: x[1])
    return results


# ---------------------------------------------------------------------------
# Metric 2 & 3: Calinski-Harabasz ratio (true vs random labels)
# ---------------------------------------------------------------------------


def calinski_harabasz_ratio(
    embeddings: np.ndarray,
    labels: list[str],
    n_random_trials: int = 30,
    seed: int = 0,
) -> dict[str, float]:
    """Compute the Calinski-Harabasz ratio: true_score / mean(random_scores).

    A ratio > 1 means the embedding separates the true label groups better
    than chance. Higher is better.

    Parameters
    ----------
    embeddings:
        (N, D) float array of cell embeddings.
    labels:
        Length-N list of cluster labels (strings).
    n_random_trials:
        Number of shuffled-label trials for the random baseline.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    Dict with keys:
        ``true``        — CH score with true labels
        ``random_mean`` — mean CH score across shuffled-label trials
        ``random_std``  — std of random CH scores
        ``ratio``       — true / random_mean (the primary reported scalar)

    Ported from ``get_true_and_random_clustering_scores()`` in pfizer/library.py.
    """
    rng = random.Random(seed)

    nan_result = {
        "true": float("nan"),
        "random_mean": float("nan"),
        "random_std": float("nan"),
        "ratio": float("nan"),
    }

    n_samples = len(labels)
    n_unique = len(set(labels))
    if n_unique < 2 or n_unique >= n_samples:
        return nan_result

    true_score = float(calinski_harabasz_score(embeddings, labels))

    random_scores: list[float] = []
    labels_list = list(labels)
    for _ in range(n_random_trials):
        shuffled = labels_list.copy()
        rng.shuffle(shuffled)
        random_scores.append(float(calinski_harabasz_score(embeddings, shuffled)))

    random_mean = float(np.mean(random_scores))
    random_std = float(np.std(random_scores))
    ratio = true_score / random_mean if random_mean > 0 else float("nan")

    return {
        "true": true_score,
        "random_mean": random_mean,
        "random_std": random_std,
        "ratio": ratio,
    }


# ---------------------------------------------------------------------------
# Metric 4a: Enrichment AUC for positive controls
# ---------------------------------------------------------------------------


def enrichment_auc(
    ranked_targets: list[str],
    positive_controls: list[str],
) -> float:
    """AUC of the enrichment curve for recovering positive-control targets.

    Walks the ranked list from best to worst reversion candidate; at each
    position records the fraction of positive controls found so far.
    AUC is computed over a normalised x-axis ∈ [0, 1].

    Returns AUC ∈ [0, 1]:
        ~0.5  random ordering
        1.0   all positive controls appear at the very top
    """
    positive_set = set(positive_controls)
    # Only count positives actually present in the ranked list
    present_positives = [p for p in positive_controls if p in set(ranked_targets)]
    n_pos = len(present_positives)
    if n_pos == 0:
        return float("nan")

    n = len(ranked_targets)
    y: list[float] = []
    running_hits = 0
    for target in ranked_targets:
        if target in positive_set:
            running_hits += 1
        y.append(running_hits / n_pos)

    x = np.linspace(0.0, 1.0, n)
    auc = float(np.trapezoid(y, x))
    return auc


# ---------------------------------------------------------------------------
# Metric 4b: Ranking agreement vs a reference list (e.g. compbio_direct)
# ---------------------------------------------------------------------------


def ranking_agreement(
    ranked_targets: list[str],
    reference_targets: list[str],
    overlap_ks: tuple[int, ...] = (10, 25),
) -> dict[str, float]:
    """Compare two ranked target lists.

    Parameters
    ----------
    ranked_targets:
        Ranked list from our method (best first).
    reference_targets:
        Reference ranked list (e.g. compbio_direct).
    overlap_ks:
        Values of k for percent-overlap-at-top-k.

    Returns
    -------
    Dict with:
        ``spearman``            — Spearman ρ on shared targets
        ``overlap_at_{k}``      — fraction of top-k that overlap, for each k
    """
    # Restrict to shared targets for Spearman
    shared = [t for t in ranked_targets if t in set(reference_targets)]
    if len(shared) < 2:
        spearman = float("nan")
    else:
        rank_a = [ranked_targets.index(t) for t in shared]
        rank_b = [reference_targets.index(t) for t in shared]
        spearman = float(spearmanr(rank_a, rank_b).statistic)

    result: dict[str, float] = {"spearman": spearman}
    for k in overlap_ks:
        top_a = set(ranked_targets[:k])
        top_b = set(reference_targets[:k])
        if len(top_b) == 0:
            result[f"overlap_at_{k}"] = float("nan")
        else:
            result[f"overlap_at_{k}"] = len(top_a & top_b) / k
    return result


# ---------------------------------------------------------------------------
# Utility: L2-normalize embeddings
# ---------------------------------------------------------------------------


def l2_norm(embeddings: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization. Wraps sklearn.preprocessing.normalize."""
    return l2_normalize(embeddings, norm="l2")
