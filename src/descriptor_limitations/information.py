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

import numpy as np
from numpy.typing import ArrayLike


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
