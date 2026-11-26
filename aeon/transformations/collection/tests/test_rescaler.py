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


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize(
    "dtype, exponent",
    [
        # the sum of squares overflows
        (np.float64, 600),
        (np.float32, 70),
        # the squares of the deviations are subnormal
        (np.float64, -536),
        (np.float32, -70),
        # the squares of the deviations underflow to 0
        (np.float64, -600),
        (np.float32, -80),
    ],
)
def test_z_norm_extreme_scales(dtype, exponent):
    """Test series at extreme scales normalise as they do at an ordinary scale."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 2, 20)).astype(dtype)
    X[1] += 100
    scale = dtype(2.0**exponent)
    normaliser = Normalizer()
    np.testing.assert_array_equal(
        normaliser._transform(X * scale), normaliser._transform(X)
    )

    X = [rng.normal(size=(2, n)).astype(dtype) for n in (15, 20, 25)]
    for xt, expected in zip(
        normaliser._transform([x * scale for x in X]), normaliser._transform(X)
    ):
        np.testing.assert_array_equal(xt, expected)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_z_norm_large_values_no_overflow_warning():
    """Test fit_transform on values beyond about 1e154 does not warn of overflow."""
    X = np.random.default_rng(0).normal(size=(4, 2, 20))
    Xt = Normalizer().fit_transform(X * 2.0**600)
    np.testing.assert_array_equal(Xt, Normalizer().fit_transform(X))


def test_z_norm_near_constant_series():
    """Test near-constant series far from zero are centred and scaled."""
    x = np.full(50, 0.1) + 1e8
    x[::7] += 1.5e-8
    Xt = Normalizer()._transform(x.reshape(1, 1, -1))
    np.testing.assert_allclose(Xt.mean(), 0, atol=1e-12)
    np.testing.assert_allclose(Xt.std(), 1)


@pytest.mark.parametrize("value", [0.0, 0.1, -3.0, 1e8 + 0.1, 2.0**600, 2.0**-600])
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
