from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy import stats as scipy_stats


@dataclass
class ConfidenceInterval:
    """Bootstrap confidence interval."""
    lower: float
    mean: float
    upper: float
    ci_level: float = 0.95

    def __str__(self):
        return f"{self.mean:.3f} [{self.lower:.3f}, {self.upper:.3f}]"


@dataclass
class PairedTestResult:
    """Result of a paired statistical test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    significant: bool = False
    alpha: float = 0.05

    def __str__(self):
        sig = "*" if self.significant else ""
        return f"{self.test_name}: stat={self.statistic:.3f}, p={self.p_value:.4f}{sig}"


def bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    statistic: str = "mean"
) -> ConfidenceInterval:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        scores: List of scores
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (0.95 = 95%)
        statistic: "mean", "median", or "std"

    Returns:
        ConfidenceInterval with lower, mean, upper bounds
    """
    if not scores:
        return ConfidenceInterval(0.0, 0.0, 0.0, ci)

    scores = np.array(scores)
    n = len(scores)

    # Bootstrap sampling
    boot_stats = []
    rng = np.random.default_rng(42)  # Reproducibility

    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        if statistic == "mean":
            boot_stats.append(np.mean(sample))
        elif statistic == "median":
            boot_stats.append(np.median(sample))
        elif statistic == "std":
            boot_stats.append(np.std(sample))
        else:
            boot_stats.append(np.mean(sample))

    boot_stats = np.array(boot_stats)

    # Percentile method
    alpha = 1 - ci
    lower = np.percentile(boot_stats, alpha / 2 * 100)
    upper = np.percentile(boot_stats, (1 - alpha / 2) * 100)

    if statistic == "mean":
        point_est = np.mean(scores)
    elif statistic == "median":
        point_est = np.median(scores)
    else:
        point_est = np.mean(scores)

    return ConfidenceInterval(lower, point_est, upper, ci)




def bootstrap_ci_paired_delta(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95
) -> ConfidenceInterval:
    """
    Compute bootstrap CI for difference in paired scores (ΔScore = B - A).

    This is the RECOMMENDED method for comparing prompt strategies on F1 scores.
    If the CI excludes 0, the difference is statistically significant.

    Args:
        scores_a: Per-contract scores for method A (e.g., P0 baseline)
        scores_b: Per-contract scores for method B (e.g., P4 SmartGuard)
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (0.95 = 95%)

    Returns:
        ConfidenceInterval for the mean difference (B - A)
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have same length")

    if not scores_a:
        return ConfidenceInterval(0.0, 0.0, 0.0, ci)

    # Compute paired differences
    deltas = np.array([b - a for a, b in zip(scores_a, scores_b)])
    n = len(deltas)
    rng = np.random.default_rng(42)

    # Bootstrap the mean difference
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(deltas, size=n, replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)

    # Percentile method
    alpha_level = 1 - ci
    lower = np.percentile(boot_means, alpha_level / 2 * 100)
    upper = np.percentile(boot_means, (1 - alpha_level / 2) * 100)
    point_est = np.mean(deltas)

    return ConfidenceInterval(lower, point_est, upper, ci)


def permutation_test(
    scores_a: List[float],
    scores_b: List[float],
    n_permutations: int = 10000,
    alpha: float = 0.05
) -> PairedTestResult:
    """
    Approximate randomization (permutation) test for paired scores.

    Tests whether the difference in mean scores is significantly different
    from what would be expected by chance. Non-parametric and makes minimal
    assumptions about the data distribution.

    Args:
        scores_a: Per-contract scores for method A
        scores_b: Per-contract scores for method B
        n_permutations: Number of random permutations
        alpha: Significance level

    Returns:
        PairedTestResult with test statistic and p-value
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have same length")

    n = len(scores_a)
    if n < 2:
        return PairedTestResult("Permutation", 0.0, 1.0, None, False, alpha)

    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    # Observed difference in means
    observed_diff = np.mean(scores_b) - np.mean(scores_a)

    # Generate null distribution by randomly swapping pairs
    rng = np.random.default_rng(42)
    null_diffs = []

    for _ in range(n_permutations):
        # For each pair, randomly decide whether to swap
        swaps = rng.choice([True, False], size=n)
        perm_a = np.where(swaps, scores_b, scores_a)
        perm_b = np.where(swaps, scores_a, scores_b)
        null_diffs.append(np.mean(perm_b) - np.mean(perm_a))

    null_diffs = np.array(null_diffs)

    # Two-tailed p-value: proportion of null diffs as extreme as observed
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))

    # Effect size: observed diff / pooled std
    pooled_std = np.std(np.concatenate([scores_a, scores_b]))
    effect_size = observed_diff / pooled_std if pooled_std > 0 else 0.0

    return PairedTestResult(
        "Permutation",
        observed_diff,  # Use diff as the "statistic"
        p_value,
        effect_size,
        p_value < alpha,
        alpha
    )




def wilcoxon_signed_rank(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05
) -> PairedTestResult:
    """
    Wilcoxon signed-rank test for paired continuous data.
    Non-parametric alternative to paired t-test.

    Args:
        scores_a: Scores for method A
        scores_b: Scores for method B
        alpha: Significance level

    Returns:
        PairedTestResult with test statistic and p-value
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have same length")

    if len(scores_a) < 5:
        return PairedTestResult("Wilcoxon", 0.0, 1.0, None, False, alpha)

    try:
        statistic, p_value = scipy_stats.wilcoxon(scores_a, scores_b)

        # Effect size: r = Z / sqrt(N)
        n = len(scores_a)
        z = scipy_stats.norm.ppf(1 - p_value / 2)
        effect_size = abs(z) / np.sqrt(n)

        return PairedTestResult(
            "Wilcoxon",
            statistic,
            p_value,
            effect_size,
            p_value < alpha,
            alpha
        )
    except Exception:
        return PairedTestResult("Wilcoxon", 0.0, 1.0, None, False, alpha)






