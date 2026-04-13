import numpy as np
import pytest

from descriptor_limitations.information import (
    BootstrapCI,
    R2CeilingResult,
    bootstrap_ci,
    conditional_entropy,
    entropy,
    mutual_information,
    r2_ceiling,
    singleton_fraction,
    within_group_variance,
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


# ---------- within_group_variance ----------

def test_within_group_variance_zero_when_groups_constant():
    """If y is constant within each group, within-group variance = 0."""
    y = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
    g = np.array([0, 0, 0, 1, 1, 1])
    for assumption in ("zero", "marginal"):
        assert within_group_variance(y, g, singleton_assumption=assumption) == pytest.approx(0.0)


def test_within_group_variance_hand_computed():
    """Two groups, no singletons.

    Group 0: [1, 3, 5], pop var = 8/3; n=3.
    Group 1: [2, 4],    pop var = 1;    n=2.
    Weighted: (3/5)*(8/3) + (2/5)*1 = 8/5 + 2/5 = 2.0.
    """
    y = np.array([1.0, 3.0, 5.0, 2.0, 4.0])
    g = np.array([0, 0, 0, 1, 1])
    assert within_group_variance(y, g, singleton_assumption="zero") == pytest.approx(2.0)


def test_within_group_variance_singletons_zero_vs_marginal():
    """Singleton group contributes 0 under 'zero', Var(y) under 'marginal'."""
    # Two samples in group 0 with values 0,2 (var=1); one singleton group with value 10.
    y = np.array([0.0, 2.0, 10.0])
    g = np.array([0, 0, 1])
    var_y = float(np.var(y, ddof=0))  # known

    # Non-singleton contribution: (2/3)*1 = 2/3
    opt = within_group_variance(y, g, singleton_assumption="zero")
    pes = within_group_variance(y, g, singleton_assumption="marginal")
    assert opt == pytest.approx(2.0 / 3.0)
    assert pes == pytest.approx(2.0 / 3.0 + (1.0 / 3.0) * var_y)


def test_within_group_variance_rejects_bad_assumption():
    y = np.array([1.0, 2.0, 3.0])
    g = np.array([0, 0, 1])
    with pytest.raises(ValueError, match="singleton_assumption"):
        within_group_variance(y, g, singleton_assumption="raise")


def test_within_group_variance_rejects_nonfinite():
    y = np.array([1.0, np.nan, 3.0])
    g = np.array([0, 0, 1])
    with pytest.raises(ValueError, match="non-finite"):
        within_group_variance(y, g, singleton_assumption="zero")


# ---------- r2_ceiling ----------

def test_r2_ceiling_perfect_when_groups_determine_y():
    """Zero within-group variance -> R^2_ceiling = 1 (no singletons, bracket collapses)."""
    y = np.array([1.0, 1.0, 5.0, 5.0, 9.0, 9.0])
    g = np.array([0, 0, 1, 1, 2, 2])
    r = r2_ceiling(y, g)
    assert isinstance(r, R2CeilingResult)
    assert r.n_singletons == 0
    assert r.optimistic == r.pessimistic
    assert r.optimistic == pytest.approx(1.0)


def test_r2_ceiling_hand_computed_no_singletons():
    """Using the same 2-group example: Var(y)_pop, wgv=2, so R^2 = 1 - 2/Var(y)."""
    y = np.array([1.0, 3.0, 5.0, 2.0, 4.0])
    g = np.array([0, 0, 0, 1, 1])
    var_y = np.var(y, ddof=0)
    expected = 1.0 - 2.0 / var_y
    r = r2_ceiling(y, g)
    assert r.n_singletons == 0
    assert r.optimistic == pytest.approx(expected)
    assert r.pessimistic == pytest.approx(expected)
    assert r.n_groups == 2
    assert r.n_samples == 5
    assert r.singleton_fraction == pytest.approx(0.0)
    assert r.var_y == pytest.approx(var_y)


def test_r2_ceiling_all_singletons_brackets_01():
    """Every group is a singleton -> optimistic=1, pessimistic=0."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    g = np.arange(5)
    r = r2_ceiling(y, g)
    assert r.n_singletons == 5
    assert r.singleton_fraction == pytest.approx(1.0)
    assert r.optimistic == pytest.approx(1.0)
    assert r.pessimistic == pytest.approx(0.0)


def test_r2_ceiling_random_groups_near_zero():
    """Random group assignment -> pessimistic close to 0 (on average)."""
    rng = np.random.default_rng(42)
    y = rng.normal(size=2000)
    g = rng.integers(0, 20, size=2000)
    r = r2_ceiling(y, g)
    assert r.n_singletons == 0 or r.n_singletons < 5
    # With N=2000 in ~20 random groups, ceiling should be near zero.
    assert abs(r.optimistic) < 0.05
    assert abs(r.pessimistic) < 0.05


def test_r2_ceiling_raises_constant_y():
    y = np.ones(10)
    g = np.arange(10)
    with pytest.raises(ValueError, match="Var\\(y\\)"):
        r2_ceiling(y, g)


def test_r2_ceiling_2d_matches_composite():
    """Multi-column groups = row-wise composite 1-D labels."""
    rng = np.random.default_rng(11)
    y = rng.normal(size=300)
    c1 = rng.integers(0, 3, size=300)
    c2 = rng.integers(0, 4, size=300)
    g_2d = np.stack([c1, c2], axis=1)
    g_1d = c1 * 10 + c2
    r_2d = r2_ceiling(y, g_2d)
    r_1d = r2_ceiling(y, g_1d)
    assert r_2d.optimistic == pytest.approx(r_1d.optimistic)
    assert r_2d.pessimistic == pytest.approx(r_1d.pessimistic)
    assert r_2d.n_groups == r_1d.n_groups


def test_r2_ceiling_mixed_singletons_bracket_orders_correctly():
    """Pessimistic <= optimistic always; strictly less when singletons exist."""
    y = np.array([0.0, 2.0, 10.0, -5.0])
    g = np.array([0, 0, 1, 2])  # group 0 has 2, groups 1,2 are singletons
    r = r2_ceiling(y, g)
    assert r.n_singletons == 2
    assert r.pessimistic < r.optimistic
    assert 0.0 <= r.pessimistic
    assert r.optimistic <= 1.0


# ---------- bootstrap_ci ----------

def _mean_of_y(y, g):
    return float(np.mean(y))


def _r2_opt(y, g):
    return r2_ceiling(y, g).optimistic


def test_bootstrap_ci_constant_func_collapses():
    """A func that ignores data returns a zero-width CI."""
    y = np.arange(20.0)
    g = np.arange(20)
    result = bootstrap_ci(
        y, g, lambda y, g: 3.14,
        n_boot=50, ci=0.95, mode="pairs", random_state=0,
    )
    assert isinstance(result, BootstrapCI)
    assert result.point_estimate == pytest.approx(3.14)
    assert result.lower == pytest.approx(3.14)
    assert result.upper == pytest.approx(3.14)


def test_bootstrap_ci_mean_covers_true_value():
    """Pairs bootstrap of the sample mean: CI covers the true mean loosely."""
    rng = np.random.default_rng(0)
    y = rng.normal(loc=5.0, scale=1.0, size=500)
    g = np.zeros(500, dtype=int)
    res = bootstrap_ci(
        y, g, _mean_of_y,
        n_boot=500, ci=0.95, mode="pairs", random_state=1,
    )
    # Sample mean SE ~ 1/sqrt(500) ~ 0.045. CI half-width ~ 0.09. True mean 5.0.
    assert res.lower < 5.0 < res.upper
    assert (res.upper - res.lower) < 0.5


def test_bootstrap_ci_r2_ceiling_pairs_brackets_point():
    """Pairs bootstrap of optimistic R^2_ceiling runs and brackets the point."""
    rng = np.random.default_rng(2)
    # Structured: 3 groups, different means + noise -> nonzero ceiling.
    g = rng.integers(0, 3, size=300)
    y = g * 5.0 + rng.normal(scale=1.0, size=300)
    res = bootstrap_ci(
        y, g, _r2_opt,
        n_boot=200, ci=0.90, mode="pairs", random_state=3,
    )
    assert res.lower <= res.point_estimate <= res.upper


def test_bootstrap_ci_within_group_zero_when_no_within_variance():
    """within-group mode: if y is constant per group, bootstrap samples are identical."""
    y = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
    g = np.array([0, 0, 0, 1, 1, 1])
    res = bootstrap_ci(
        y, g, _r2_opt,
        n_boot=100, ci=0.95, mode="within-group", random_state=4,
    )
    assert res.lower == pytest.approx(res.upper)
    assert res.point_estimate == pytest.approx(1.0)


def test_bootstrap_ci_reproducible():
    """Same random_state -> identical samples."""
    y = np.arange(30.0)
    g = np.tile(np.arange(3), 10)
    r1 = bootstrap_ci(y, g, _mean_of_y, n_boot=50, ci=0.95, mode="pairs", random_state=7)
    r2 = bootstrap_ci(y, g, _mean_of_y, n_boot=50, ci=0.95, mode="pairs", random_state=7)
    assert np.allclose(r1.samples, r2.samples)


def test_bootstrap_ci_rejects_bad_mode():
    with pytest.raises(ValueError, match="mode"):
        bootstrap_ci(
            np.array([1.0, 2.0]), np.array([0, 1]), _mean_of_y,
            n_boot=10, ci=0.95, mode="jackknife", random_state=0,
        )


def test_bootstrap_ci_rejects_bad_ci():
    with pytest.raises(ValueError, match="ci"):
        bootstrap_ci(
            np.array([1.0, 2.0]), np.array([0, 1]), _mean_of_y,
            n_boot=10, ci=1.5, mode="pairs", random_state=0,
        )


def test_bootstrap_ci_rejects_bad_n_boot():
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_ci(
            np.array([1.0, 2.0]), np.array([0, 1]), _mean_of_y,
            n_boot=0, ci=0.95, mode="pairs", random_state=0,
        )


def test_bootstrap_ci_catches_nonfinite():
    def bad_func(y, g):
        return np.nan

    with pytest.raises(ValueError, match="non-finite"):
        bootstrap_ci(
            np.array([1.0, 2.0]), np.array([0, 1]), bad_func,
            n_boot=10, ci=0.95, mode="pairs", random_state=0,
        )
