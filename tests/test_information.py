import numpy as np
import pytest

from descriptor_limitations.information import entropy


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
