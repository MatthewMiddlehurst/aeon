"""Tests for the rescaling transformers."""

import numpy as np
import pytest

from aeon.transformations.collection._rescale import Centerer, MinMaxScaler, Normalizer


def test_z_norm():
    """Test the Normalize class.

    This function creates a 3D numpy array, applies z-normalization using the
    Normalise class, and asserts that the transformed data has a mean close to 0 and a
    standard deviation close to 1 along the specified axis.
    """
    X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    normaliser = Normalizer()
    X_transformed = normaliser._transform(X)

    mean = np.mean(X_transformed, axis=-1)
    std = np.std(X_transformed, axis=-1)

    assert np.allclose(mean, 0)
    assert np.allclose(std, 1)


def _numpy_z_norm(X):
    """Z-normalise with numpy's mean and std, the result for ordinary series."""
    with np.errstate(all="ignore"):
        return (X - X.mean(axis=-1, keepdims=True)) / X.std(axis=-1, keepdims=True)


def test_z_norm_ordinary_series_unchanged():
    """Test ordinary series are normalised with numpy's mean and std, bit for bit."""
    X = np.random.default_rng(0).normal(5, 3, size=(5, 2, 30))
    np.testing.assert_array_equal(Normalizer()._transform(X), _numpy_z_norm(X))


@pytest.mark.parametrize("value", [0.0, 0.1, 1 / 3, -3.0, 1e8 + 0.1])
def test_z_norm_constant_series(value):
    """Test constant series normalise to zeros, without changing other series."""
    X = np.random.default_rng(0).normal(size=(3, 2, 50))
    X[1, 0] = value
    expected = _numpy_z_norm(X)
    expected[1, 0] = 0
    np.testing.assert_array_equal(Normalizer()._transform(X), expected)

    X = [X[0], X[1, :, :40], X[2, :, :30]]
    expected = [_numpy_z_norm(x) for x in X]
    expected[1][0] = 0
    for xt, e in zip(Normalizer()._transform(X), expected):
        np.testing.assert_array_equal(xt, e)


def test_centering():
    """Test the Centerer class."""
    X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    std = Centerer()
    X_transformed = std._transform(X)

    mean = np.mean(X_transformed, axis=-1)

    assert np.allclose(mean, 0)


def test_min_max():
    """Test the MinMaxScaler class."""
    X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    minmax = MinMaxScaler()
    X_transformed = minmax._transform(X)

    min_val = np.min(X_transformed, axis=-1)
    max_val = np.max(X_transformed, axis=-1)

    assert np.allclose(min_val, 0)
    assert np.allclose(max_val, 1)
    with pytest.raises(ValueError, match="should be less than max value"):
        minmax = MinMaxScaler(min=1, max=0)
        X_transformed = minmax._transform(X)
