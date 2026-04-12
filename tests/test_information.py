import numpy as np
import pytest

from descriptor_limitations.information import (
    conditional_entropy,
    entropy,
    mutual_information,
    singleton_fraction,
)


def test_entropy_uniform_binary():
    """Fair coin: H = 1 bit."""
    assert entropy([0.5, 0.5]) == pytest.approx(1.0)


def test_entropy_uniform_k():
    """Uniform on k atoms: H = log2(k)."""
    for k in [2, 3, 4, 8, 16]:
        p = np.full(k, 1.0 / k)
        assert entropy(p) == pytest.approx(np.log2(k))


def test_entropy_delta_is_zero():
    """Degenerate distribution: H = 0."""
    assert entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert entropy([0.0, 1.0]) == pytest.approx(0.0)


def test_entropy_known_value():
    """Biased coin p=0.25: H = -0.25 log2(0.25) - 0.75 log2(0.75)."""
    assert entropy([0.25, 0.75]) == pytest.approx(0.8112781244591328)


def test_entropy_rejects_negative():
    with pytest.raises(ValueError, match="negative"):
        entropy([-0.1, 1.1])


def test_entropy_rejects_unnormalized():
    with pytest.raises(ValueError, match="sum to 1"):
        entropy([0.3, 0.3])
    with pytest.raises(ValueError, match="sum to 1"):
        entropy([0.5, 0.6])


def test_entropy_rejects_wrong_shape():
    with pytest.raises(ValueError, match="1-D"):
        entropy([[0.5, 0.5], [0.5, 0.5]])


# ---------- conditional_entropy ----------

def test_conditional_entropy_deterministic():
    """If X determines y, H(X_math|Y_math) = 0 (plug-in; Miller-Madow adds 0
    because each stratum has K_j = 1)."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 5, size=200)
    y = X * 2  # deterministic function of X
    assert conditional_entropy(y, X, correction="none") == pytest.approx(0.0)
    assert conditional_entropy(y, X, correction="miller-madow") == pytest.approx(0.0)


def test_conditional_entropy_independent_constant_X():
    """Constant descriptor: H(X_math|Y_math) = H(X_math)."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, size=2000)
    X = np.zeros(2000, dtype=int)
    # Plug-in should equal plug-in entropy of y.
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    H_y = -np.sum(p * np.log2(p))
    assert conditional_entropy(y, X, correction="none") == pytest.approx(H_y)


def test_conditional_entropy_known_2x2():
    """Hand-computed 2x2 joint.

    Joint counts:
        y=0  y=1
    X=0  30   10    (n_.0 = 40)
    X=1  20   40    (n_.1 = 60)
    N=100. Within X=0: p=(3/4,1/4) -> H=0.8112781...; weight 0.4.
    Within X=1: p=(1/3,2/3) -> H=0.9182958...; weight 0.6.
    H(X_math|Y_math) = 0.4*0.8112781 + 0.6*0.9182958 = 0.8754887...
    """
    y = np.array([0]*30 + [1]*10 + [0]*20 + [1]*40)
    X = np.array([0]*40 + [1]*60)
    expected = 0.4 * 0.8112781244591328 + 0.6 * 0.9182958340544896
    assert conditional_entropy(y, X, correction="none") == pytest.approx(expected)


def test_miller_madow_magnitude():
    """Correction = sum_j (K_j - 1) / (2 N ln 2).

    2 strata, K_0 = K_1 = 3, N = 60 -> correction = (2+2)/(2*60*ln2)
    = 4/(120*ln2) bits.
    """
    y = np.array([0]*10 + [1]*10 + [2]*10 + [0]*10 + [1]*10 + [2]*10)
    X = np.array([0]*30 + [1]*30)
    plugin = conditional_entropy(y, X, correction="none")
    mm = conditional_entropy(y, X, correction="miller-madow")
    expected_delta = 4.0 / (2.0 * 60.0 * np.log(2))
    assert (mm - plugin) == pytest.approx(expected_delta)


def test_conditional_entropy_multicolumn_X_matches_composite():
    """Two binary columns = one 4-valued column under joint combination."""
    rng = np.random.default_rng(2)
    col1 = rng.integers(0, 2, size=500)
    col2 = rng.integers(0, 2, size=500)
    y = rng.integers(0, 3, size=500)
    X_2d = np.stack([col1, col2], axis=1)
    X_1d_composite = col1 * 2 + col2  # 0..3
    h_2d = conditional_entropy(y, X_2d, correction="none")
    h_1d = conditional_entropy(y, X_1d_composite, correction="none")
    assert h_2d == pytest.approx(h_1d)


def test_conditional_entropy_length_mismatch():
    with pytest.raises(ValueError, match="samples"):
        conditional_entropy(np.arange(10), np.arange(9))


def test_conditional_entropy_empty():
    with pytest.raises(ValueError, match="empty"):
        conditional_entropy(np.array([]), np.array([]))


def test_conditional_entropy_bad_correction():
    with pytest.raises(ValueError, match="correction"):
        conditional_entropy([0, 1], [0, 1], correction="chao-shen")


# ---------- singleton_fraction ----------

def test_singleton_fraction_none():
    X = np.array([0, 0, 1, 1, 2, 2])
    assert singleton_fraction(X) == pytest.approx(0.0)


def test_singleton_fraction_all():
    X = np.arange(10)
    assert singleton_fraction(X) == pytest.approx(1.0)


def test_singleton_fraction_mixed():
    X = np.array([0, 0, 0, 1, 2, 3])  # 3 singletons out of 6
    assert singleton_fraction(X) == pytest.approx(0.5)


def test_singleton_fraction_2d():
    X = np.array([[0, 0], [0, 0], [1, 1], [1, 2]])  # (0,0) x2; (1,1), (1,2) singletons
    assert singleton_fraction(X) == pytest.approx(0.5)


# ---------- mutual_information ----------

def test_mutual_information_self():
    """I(y; y) = H(y) (plug-in)."""
    rng = np.random.default_rng(3)
    y = rng.integers(0, 5, size=500)
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    H_y = -np.sum(p * np.log2(p))
    assert mutual_information(y, y, correction="none") == pytest.approx(H_y)


def test_mutual_information_deterministic():
    """y = f(X), invertible -> I(y; X) = H(y) (plug-in)."""
    rng = np.random.default_rng(4)
    X = rng.integers(0, 4, size=800)
    y = 3 * X + 1
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    H_y = -np.sum(p * np.log2(p))
    assert mutual_information(y, X, correction="none") == pytest.approx(H_y)


def test_mutual_information_symmetry_plugin():
    """I(y; X) == I(X; y) under plug-in (no MM asymmetry)."""
    rng = np.random.default_rng(5)
    y = rng.integers(0, 3, size=600)
    X = rng.integers(0, 4, size=600)
    a = mutual_information(y, X, correction="none")
    b = mutual_information(X, y, correction="none")
    assert a == pytest.approx(b)


def test_mutual_information_independent_small_for_large_n():
    """Independent y, X -> plug-in I is small but positive (finite-sample bias)."""
    rng = np.random.default_rng(6)
    n = 5000
    y = rng.integers(0, 3, size=n)
    X = rng.integers(0, 3, size=n)
    I_plugin = mutual_information(y, X, correction="none")
    assert 0.0 <= I_plugin < 0.05


def test_mutual_information_nonnegative_plugin():
    """Plug-in I is always >= 0."""
    rng = np.random.default_rng(7)
    for _ in range(10):
        y = rng.integers(0, 4, size=200)
        X = rng.integers(0, 4, size=200)
        assert mutual_information(y, X, correction="none") >= 0.0
