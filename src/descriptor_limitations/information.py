"""Information-theoretic primitives for descriptor-limited prediction bounds.

All entropies are in bits (log base 2).

Convention
----------
Throughout this module, code uses ML naming:
    `y` : outcome (what we predict)          -- math symbol X
    `X` : descriptors (what we predict from) -- math symbol Y

Docstrings state results using the information-theoretic math symbols
(H(X), H(X|Y), I(X;Y)) and map them back to the code arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike

_LN2 = np.log(2.0)


@dataclass(frozen=True)
class R2CeilingResult:
    """Bracketed R^2 ceiling with diagnostics for singleton-group ambiguity.

    Attributes
    ----------
    optimistic : float
        Highest R^2_ceiling consistent with the data. Computed by assigning
        Var(y | g=j) = 0 to every singleton group (g_j with n_j = 1).
        Use as the *upper bound* on R^2_ceiling.
    pessimistic : float
        Lowest R^2_ceiling consistent with the data. Computed by assigning
        Var(y | g=j) = Var(y) to every singleton group. Use as the *lower
        bound* on R^2_ceiling.
    n_samples : int
        Total samples used.
    n_groups : int
        Number of distinct group labels (joint of all columns if 2-D).
    n_singletons : int
        Number of singleton groups (n_j = 1). When 0,
        `optimistic == pessimistic` exactly and either is the ceiling.
    singleton_fraction : float
        Fraction of samples that fall into singleton groups, in [0, 1].
        High values indicate the descriptor is too high-resolution for
        the dataset; consider coarsening.
    var_y : float
        Population variance of `y` (ddof=0). Provided for context and so
        the within-group variance can be reconstructed if desired.
    """

    optimistic: float
    pessimistic: float
    n_samples: int
    n_groups: int
    n_singletons: int
    singleton_fraction: float
    var_y: float


def entropy(p: ArrayLike) -> float:
    """Shannon entropy H(X) of a discrete distribution, in bits.

    Computes H(X) = -sum_i p_i * log2(p_i), with the convention
    0 * log2(0) = 0.

    Parameters
    ----------
    p : array-like, shape (k,)
        Probability vector over k outcomes of the random variable X.
        Must be non-negative and sum to 1 (within tolerance 1e-8).

    Returns
    -------
    H : float
        Entropy in bits. Lies in [0, log2(k)].

    Raises
    ------
    ValueError
        If `p` is not 1-D, has negative entries, or does not sum to 1.

    Notes
    -----
    This is the distribution-level entropy. For entropy estimated from
    samples (with bias correction), use the sample-based estimators built
    on top of this primitive (e.g. `conditional_entropy`).
    """
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"p must be 1-D, got shape {p.shape}")
    if np.any(p < 0):
        raise ValueError("p contains negative entries")
    total = p.sum()
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f"p must sum to 1, got {total}")
    nz = p > 0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def _as_composite_labels(X: ArrayLike, n_expected: int) -> np.ndarray:
    """Reduce `X` (1-D or 2-D) to a 1-D array of composite labels.

    For 2-D input of shape (n, d), each row is combined into a single
    hashable tuple so that groupings are by the joint value of all
    descriptor columns. For 1-D input, the array is returned unchanged
    (as an object array for uniform downstream handling).
    """
    arr = np.asarray(X)
    if arr.ndim not in (1, 2):
        raise ValueError(f"X must be 1-D or 2-D, got shape {arr.shape}")
    if arr.shape[0] != n_expected:
        raise ValueError(
            f"X has {arr.shape[0]} samples, expected {n_expected}"
        )
    if arr.ndim == 1:
        # Return an integer-encoded view so downstream np.unique is cheap
        # and deterministic regardless of the original dtype.
        _, inv = np.unique(arr, return_inverse=True)
        return inv
    # 2-D: integer-encode each column independently, then pack rows into
    # a single composite integer label. This avoids object-array pitfalls
    # in np.unique and keeps the encoding collision-free.
    n, d = arr.shape
    cols_encoded = np.empty((n, d), dtype=np.int64)
    for c in range(d):
        _, inv = np.unique(arr[:, c], return_inverse=True)
        cols_encoded[:, c] = inv
    # Assign a unique integer to each distinct row.
    _, row_labels = np.unique(cols_encoded, axis=0, return_inverse=True)
    return row_labels


def singleton_fraction(X: ArrayLike) -> float:
    """Fraction of samples that fall into singleton strata of `X`.

    A singleton stratum is a unique descriptor value observed exactly
    once. High singleton fraction indicates the conditional entropy
    H(X_math | Y_math) estimate is unreliable: each singleton contributes
    zero to the plug-in estimate (since p(x|y)=1 there), but that is an
    artifact of finite sampling rather than genuine determinism.

    Parameters
    ----------
    X : array-like, shape (n,) or (n, d)
        Descriptor samples. 2-D input is collapsed row-wise to composite
        labels (see `_as_composite_labels`).

    Returns
    -------
    frac : float
        Number of samples in singleton strata divided by n. In [0, 1].
    """
    arr = np.asarray(X)
    if arr.shape[0] == 0:
        raise ValueError("X is empty")
    labels = _as_composite_labels(X, n_expected=arr.shape[0])
    _, counts = np.unique(labels, return_counts=True)
    n = labels.shape[0]
    return float(counts[counts == 1].sum() / n)


def conditional_entropy(
    y: ArrayLike,
    X: ArrayLike,
    correction: Literal["none", "miller-madow"] = "miller-madow",
) -> float:
    """Conditional entropy H(X_math | Y_math) in bits, estimated from samples.

    Math
    ----
    H(X | Y) = sum_y p(y) * H(X | Y=y)
            ~= -sum_{i,j} (n_ij / N) * log2(n_ij / n_.j)

    with the convention 0 * log2(0) = 0. With the Miller-Madow bias
    correction (recommended default), the plug-in estimate above is
    increased by

        sum_j (K_j - 1) / (2 N ln 2)

    where K_j is the number of distinct outcomes observed in stratum
    Y_math = y_j. This reduces the finite-sample downward bias from
    O(1/N) to O(1/N^2).

    Convention
    ----------
    Code `y` (shape (n,))             = math X (outcome).
    Code `X` (shape (n,) or (n, d))   = math Y (descriptor).
    Multi-column `X` is combined row-wise into composite labels, so the
    result is H(X_math | joint of all descriptor columns).

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome labels (any hashable dtype).
    X : array-like, shape (n,) or (n, d)
        Descriptor labels (any hashable dtype).
    correction : {"none", "miller-madow"}, default "miller-madow"
        Bias correction for the plug-in estimator.

    Returns
    -------
    H_cond : float
        Estimated H(X_math | Y_math) in bits. Non-negative.

    Raises
    ------
    ValueError
        If lengths disagree, inputs are empty, or `correction` is unknown.

    Notes
    -----
    Singleton strata (descriptor values seen exactly once) contribute 0
    to the plug-in term, which biases H(X|Y) downward when the descriptor
    is high-cardinality relative to n. Use `singleton_fraction(X)` as a
    diagnostic; if large, prefer a coarser descriptor or report bounds
    explicitly.
    """
    if correction not in ("none", "miller-madow"):
        raise ValueError(
            f"correction must be 'none' or 'miller-madow', got {correction!r}"
        )
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y_arr.shape}")
    n = y_arr.shape[0]
    if n == 0:
        raise ValueError("y is empty")
    X_labels = _as_composite_labels(X, n_expected=n)

    # Plug-in H(X|Y) = -(1/N) sum_{i,j} n_ij log2(n_ij / n_.j)
    #               = -(1/N) [sum_{i,j} n_ij log2(n_ij) - sum_j n_.j log2(n_.j)]
    total = 0.0
    mm_correction = 0.0
    unique_x, x_inv = np.unique(X_labels, return_inverse=True)
    for j in range(unique_x.shape[0]):
        mask = x_inv == j
        n_j = int(mask.sum())
        if n_j == 0:
            continue
        _, counts = np.unique(y_arr[mask], return_counts=True)
        # Conditional entropy within stratum j, times p(y_j) = n_j / N.
        # p(x|y_j) = counts / n_j
        p_cond = counts / n_j
        h_j = -np.sum(p_cond * np.log2(p_cond))  # zero counts never appear
        total += (n_j / n) * h_j
        if correction == "miller-madow":
            K_j = counts.shape[0]  # observed support size in stratum j
            mm_correction += (K_j - 1) / (2.0 * n * _LN2)

    return float(total + mm_correction)


def _marginal_entropy(
    y: ArrayLike,
    correction: Literal["none", "miller-madow"] = "miller-madow",
) -> float:
    """Sample estimate of H(X_math) from labels `y`, with optional MM correction.

    Miller-Madow correction is (K - 1) / (2 N ln 2), where K is the number
    of distinct labels observed in `y`.
    """
    if correction not in ("none", "miller-madow"):
        raise ValueError(
            f"correction must be 'none' or 'miller-madow', got {correction!r}"
        )
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y_arr.shape}")
    n = y_arr.shape[0]
    if n == 0:
        raise ValueError("y is empty")
    _, counts = np.unique(y_arr, return_counts=True)
    p = counts / n
    H_plugin = -np.sum(p * np.log2(p))
    if correction == "miller-madow":
        K = counts.shape[0]
        H_plugin += (K - 1) / (2.0 * n * _LN2)
    return float(H_plugin)


def mutual_information(
    y: ArrayLike,
    X: ArrayLike,
    correction: Literal["none", "miller-madow"] = "miller-madow",
) -> float:
    """Mutual information I(X_math; Y_math) in bits, estimated from samples.

    Math
    ----
    I(X; Y) = H(X) - H(X | Y).

    Convention
    ----------
    Code `y` (shape (n,))           = math X (outcome).
    Code `X` (shape (n,) or (n, d)) = math Y (descriptor).

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome labels.
    X : array-like, shape (n,) or (n, d)
        Descriptor labels; 2-D is combined row-wise.
    correction : {"none", "miller-madow"}, default "miller-madow"
        Applied to both H(X_math) and H(X_math | Y_math) consistently.

    Returns
    -------
    I : float
        Mutual information in bits. Theoretically non-negative; the
        plug-in estimate is also non-negative. The Miller-Madow
        estimate can be slightly negative in degenerate small-sample
        cases because the correction terms for H(X) and H(X|Y) are
        not linked.

    Notes
    -----
    For the true quantity, I(X;Y) = I(Y;X). The plug-in estimate is
    symmetric up to floating-point error; the Miller-Madow estimate is
    NOT symmetric because the correction uses per-stratum support sizes
    of the conditioning variable.
    """
    H_marginal = _marginal_entropy(y, correction=correction)
    H_cond = conditional_entropy(y, X, correction=correction)
    return H_marginal - H_cond


def _prepare_group_labels(
    y: ArrayLike, groups: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Validate (y, groups) and return (y_float, integer group labels)."""
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y_arr.shape}")
    n = y_arr.shape[0]
    if n == 0:
        raise ValueError("y is empty")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("y contains non-finite values")
    labels = _as_composite_labels(groups, n_expected=n)
    return y_arr, labels


def within_group_variance(
    y: ArrayLike,
    groups: ArrayLike,
    singleton_assumption: Literal["zero", "marginal"],
) -> float:
    """Weighted within-group variance E_g[Var(y | g)] (population, ddof=0).

    Math
    ----
    E[Var(y | g)] = sum_j (n_j / N) * Var(y_j)
    where Var(y_j) is the population variance of y within group j. For
    singleton groups (n_j = 1), Var(y_j) is undefined; the user must
    declare an assumption explicitly.

    Parameters
    ----------
    y : array-like, shape (n,)
        Continuous outcome.
    groups : array-like, shape (n,) or (n, d)
        Group labels. 2-D is combined row-wise into composite labels.
    singleton_assumption : {"zero", "marginal"}
        How to fill in Var(y | g=j) for singleton groups:
          * "zero"     -> Var(y_j) = 0 (optimistic; yields the upper
            bound on R^2_ceiling).
          * "marginal" -> Var(y_j) = Var(y) (pessimistic; yields the
            lower bound on R^2_ceiling).

    Returns
    -------
    wgv : float
        Weighted within-group variance in the same units as y^2.

    Raises
    ------
    ValueError
        If inputs are malformed or `singleton_assumption` is unknown.
    """
    if singleton_assumption not in ("zero", "marginal"):
        raise ValueError(
            "singleton_assumption must be 'zero' or 'marginal', "
            f"got {singleton_assumption!r}"
        )
    y_arr, labels = _prepare_group_labels(y, groups)
    n = y_arr.shape[0]
    unique, inv = np.unique(labels, return_inverse=True)
    var_y = float(np.var(y_arr, ddof=0))
    total = 0.0
    for j in range(unique.shape[0]):
        mask = inv == j
        n_j = int(mask.sum())
        if n_j == 1:
            v_j = 0.0 if singleton_assumption == "zero" else var_y
        else:
            v_j = float(np.var(y_arr[mask], ddof=0))
        total += (n_j / n) * v_j
    return total


def r2_ceiling(
    y: ArrayLike,
    groups: ArrayLike,
) -> R2CeilingResult:
    """Information-theoretic R^2 ceiling for predicting `y` from `groups`.

    Math
    ----
    R^2_ceiling = 1 - Var(y | g) / Var(y)

    where Var(y) is the marginal population variance (ddof=0) and
    Var(y | g) = E_g[Var(y | g=j)]. This is an upper bound on the
    coefficient of determination achievable by any predictor that sees
    only `groups`, by the law of total variance.

    Convention
    ----------
    Code `y`      (shape (n,))           = math X (continuous outcome).
    Code `groups` (shape (n,) or (n, d)) = math Y (categorical descriptor;
    columns are combined row-wise into a joint label).

    Singleton groups (n_j = 1) carry zero information about Var(y | g=j).
    The function reports both the optimistic ceiling (singletons -> Var=0)
    and the pessimistic ceiling (singletons -> Var=Var(y)). When there are
    no singletons, the two numbers coincide exactly.

    Parameters
    ----------
    y : array-like, shape (n,)
        Continuous outcome. Must be finite.
    groups : array-like, shape (n,) or (n, d)
        Group labels, 1-D or row-wise-composite 2-D.

    Returns
    -------
    result : R2CeilingResult
        Bracketed ceiling with singleton diagnostics.

    Raises
    ------
    ValueError
        If inputs are malformed, or if Var(y) = 0 (R^2 undefined).
    """
    y_arr, labels = _prepare_group_labels(y, groups)
    n = y_arr.shape[0]
    var_y = float(np.var(y_arr, ddof=0))
    if var_y == 0.0:
        raise ValueError("Var(y) = 0: R^2_ceiling is undefined (y is constant)")

    unique, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    n_groups = unique.shape[0]
    n_singletons = int((counts == 1).sum())
    n_in_singletons = n_singletons  # each singleton group contributes 1 sample

    # Non-singleton contribution is shared by both bounds.
    non_singleton_contrib = 0.0
    for j in range(n_groups):
        if counts[j] == 1:
            continue
        v_j = float(np.var(y_arr[inv == j], ddof=0))
        non_singleton_contrib += (counts[j] / n) * v_j

    # Optimistic: singletons contribute 0.
    wgv_opt = non_singleton_contrib
    # Pessimistic: singletons contribute Var(y) each, weighted by 1/N each.
    wgv_pes = non_singleton_contrib + (n_singletons / n) * var_y

    optimistic = 1.0 - wgv_opt / var_y
    pessimistic = 1.0 - wgv_pes / var_y

    # Clip float noise only; values outside [0, 1] on real data indicate
    # a bug and should be preserved for inspection rather than silently hidden.
    if -1e-12 < optimistic - 1.0 < 0:
        optimistic = min(optimistic, 1.0)
    if -1e-12 < -pessimistic < 0:
        pessimistic = max(pessimistic, 0.0)

    return R2CeilingResult(
        optimistic=float(optimistic),
        pessimistic=float(pessimistic),
        n_samples=n,
        n_groups=n_groups,
        n_singletons=n_singletons,
        singleton_fraction=n_in_singletons / n,
        var_y=var_y,
    )


@dataclass(frozen=True)
class BootstrapCI:
    """Percentile-based bootstrap confidence interval for a scalar statistic.

    Attributes
    ----------
    point_estimate : float
        Value of `func(y, groups)` on the original (non-resampled) data.
    lower, upper : float
        Percentile bounds at the requested confidence level.
    ci_level : float
        Confidence level used (e.g. 0.95).
    n_boot : int
        Number of bootstrap replicates.
    mode : str
        Resampling scheme ("pairs" or "within-group").
    samples : np.ndarray, shape (n_boot,)
        Raw bootstrap replicates of the statistic, for diagnostics
        (histogram, bias estimate, etc.).
    """

    point_estimate: float
    lower: float
    upper: float
    ci_level: float
    n_boot: int
    mode: str
    samples: np.ndarray = field(repr=False)


def bootstrap_ci(
    y: ArrayLike,
    groups: ArrayLike,
    func: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int,
    ci: float,
    mode: Literal["pairs", "within-group"],
    random_state: int | np.random.Generator | None = None,
) -> BootstrapCI:
    """Percentile bootstrap CI for a scalar statistic of (y, groups).

    Two resampling schemes are supported; the choice must be made
    explicitly because it reflects an inferential claim:

    * "pairs": resample (y_i, groups_i) tuples with replacement. Use
      when the data are i.i.d. draws from a population (observational
      or randomly designed experiments). Captures both within-group
      and between-group variability.
    * "within-group": for each distinct group label, resample that
      group's y values with replacement; group structure is fixed.
      Use when groups are a fixed design and only measurement noise
      within groups is random. Yields zero CI for groups with no
      replicates.

    For fixed full-factorial designs with no replicates (e.g. Doyle
    Buchwald-Hartwig), neither scheme is fully coherent; prefer the
    singleton bracket from `r2_ceiling` as the uncertainty summary.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome passed through to `func`.
    groups : array-like, shape (n,) or (n, d)
        Groups passed through to `func`.
    func : callable
        Must accept `(y_resampled, groups_resampled)` and return a
        finite float.
    n_boot : int
        Number of bootstrap replicates (required). Typical 1000.
    ci : float
        Confidence level in (0, 1), e.g. 0.95.
    mode : {"pairs", "within-group"}
        Resampling scheme.
    random_state : int, np.random.Generator, or None
        Seed or generator for reproducibility.

    Returns
    -------
    BootstrapCI
        Bracketed CI with the raw replicates attached.

    Raises
    ------
    ValueError
        If `n_boot` or `ci` is out of range, `mode` is unknown, or a
        bootstrap replicate yields a non-finite value.
    """
    if mode not in ("pairs", "within-group"):
        raise ValueError(
            f"mode must be 'pairs' or 'within-group', got {mode!r}"
        )
    if not (isinstance(n_boot, (int, np.integer)) and n_boot >= 1):
        raise ValueError(f"n_boot must be a positive int, got {n_boot!r}")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci!r}")

    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)
    n = y_arr.shape[0]
    if n == 0:
        raise ValueError("y is empty")
    if groups_arr.shape[0] != n:
        raise ValueError(
            f"groups has {groups_arr.shape[0]} samples, expected {n}"
        )

    rng = np.random.default_rng(random_state)
    point = float(func(y_arr, groups_arr))
    if not np.isfinite(point):
        raise ValueError("func returned a non-finite value on original data")

    samples = np.empty(n_boot, dtype=float)

    if mode == "pairs":
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            val = float(func(y_arr[idx], groups_arr[idx]))
            if not np.isfinite(val):
                raise ValueError(
                    f"func returned non-finite on bootstrap replicate {b}"
                )
            samples[b] = val
    else:  # within-group
        # Group samples are resampled with replacement *within* each group;
        # 2-D groups are handled by composite labels.
        labels = _as_composite_labels(groups_arr, n_expected=n)
        unique, inv = np.unique(labels, return_inverse=True)
        # Precompute indices per group.
        group_indices = [np.where(inv == j)[0] for j in range(unique.shape[0])]
        for b in range(n_boot):
            resampled = np.empty(n, dtype=np.int64)
            for g_idx in group_indices:
                picks = rng.integers(0, g_idx.shape[0], size=g_idx.shape[0])
                resampled[g_idx] = g_idx[picks]
            val = float(func(y_arr[resampled], groups_arr[resampled]))
            if not np.isfinite(val):
                raise ValueError(
                    f"func returned non-finite on bootstrap replicate {b}"
                )
            samples[b] = val

    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(samples, alpha))
    upper = float(np.quantile(samples, 1.0 - alpha))
    return BootstrapCI(
        point_estimate=point,
        lower=lower,
        upper=upper,
        ci_level=ci,
        n_boot=n_boot,
        mode=mode,
        samples=samples,
    )
