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

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

_LN2 = np.log(2.0)


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
