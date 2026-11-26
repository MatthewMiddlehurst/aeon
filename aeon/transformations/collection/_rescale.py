"""Rescaler classes for z-normalization, centering and min-max scaling."""

import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer


class Normalizer(BaseCollectionTransformer):
    """Normaliser transformer for collections.

    This transformer applies z-normalization  applied along the timepoints axis (the
    last axis). For multivariate data, it normalizes each channel independently.
    A series with a standard deviation of 0 is only centred, so a constant series
    normalises to zeros.

    Examples
    --------
    >>> from aeon.transformations.collection import Normalizer
    >>> import numpy as np
    >>> X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    >>> normaliser = Normalizer()
    >>> Xt = normaliser.fit_transform(X)
    >>> mean=np.mean(Xt, axis=-1)
    >>> std = np.std(Xt, axis=-1)
    >>> assert np.allclose(mean, 0)
    >>> assert np.allclose(std, 1)
    """

    _tags = {
        "X_inner_type": ["numpy3D", "np-list"],
        "fit_is_empty": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
    }

    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None) -> np.ndarray:
        """
        Transform method to apply the normalisation.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to transform. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        y : None
            Ignored.

        Returns
        -------
        np.ndarray or list
            The normalized data.
        """
        if isinstance(X, np.ndarray):
            # Case 1: X is a single 3D array
            return _z_normalise(X)
        # Case 2: X is a list of 2D arrays
        return [_z_normalise(x) for x in X]


def _z_normalise(X: np.ndarray) -> np.ndarray:
    """Z-normalise each series along the last axis of an array.

    Ordinary series are normalised with numpy's mean and standard deviation. Series
    for which those are inaccurate are recomputed by ``_z_normalise_unreliable``, see
    ``_stats_unreliable``.
    """
    with np.errstate(over="ignore"):
        # the overflow of large values is handled by _z_normalise_unreliable
        mean = np.mean(X, axis=-1, keepdims=True)
        std = np.std(X, axis=-1, keepdims=True)
    unreliable = _stats_unreliable(X.shape[-1], mean, std)
    # unreliable series are left as a float copy here, and recomputed below
    Xt = (X - np.where(unreliable, 0, mean)) / np.where(unreliable, 1, std)
    for idx in zip(*np.nonzero(unreliable[..., 0])):
        Xt[idx] = _z_normalise_unreliable(Xt[idx])
    return Xt


def _stats_unreliable(n: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Return where a series' mean and standard deviation are inaccurate.

    - Values beyond about 1e154 overflow the sum of squares, and values near the
      largest float the sum itself, which gives a mean or std of inf or NaN.
    - The squares of deviations below about 1e-154 are subnormal or underflow to 0,
      which loses precision or gives a std of 0.
    - When the std is within rounding of the mean, e.g. for a near-constant series
      far from zero, the mean's rounding error can exceed the variation of the
      series. This includes a std of 0.
    """
    finfo = np.finfo(std.dtype)
    return (
        ~(np.isfinite(mean) & np.isfinite(std))
        | (std < np.sqrt(n * finfo.smallest_normal))
        | (std <= n * finfo.eps * np.abs(mean))
    )


def _z_normalise_unreliable(x: np.ndarray) -> np.ndarray:
    """Z-normalise a 1D float series whose statistics are inaccurate.

    The series is first scaled by the power of two that brings its largest absolute
    value into [0.5, 1). This is exact, and its statistics then neither overflow
    nor underflow, so a series that only had a problem of scale normalises to the
    same values as at an ordinary scale. A series whose std is still within
    rounding of its mean is centred a second time, and a constant series
    normalises to zeros.
    """
    max_abs = np.max(np.abs(x))
    if not np.isfinite(max_abs):
        # a series with inf or NaN values normalises to NaN, as numpy's statistics do
        return np.full_like(x, np.nan)
    y = np.ldexp(x, -np.frexp(max_abs)[1])
    mean = np.mean(y)
    std = np.std(y)
    if std > y.shape[0] * np.finfo(y.dtype).eps * abs(mean):
        return (y - mean) / std
    if y.min() == y.max():
        return np.zeros_like(y)
    y -= mean
    y -= np.mean(y)
    std = np.std(y)
    return y / std if std > 0 else y


class MinMaxScaler(BaseCollectionTransformer):
    """MinMax transformer for collections.

    This transformer scales a collection of time series data to a specified range
    [min,max] along the time axis. The default is to [0,1]. For multivariate,
    it scales each channel independently.


    Parameters
    ----------
    min: float, default=0
        Minimum value of the range to scale to.
    max: float, default=1
        Maximum value of the range to scale to.

    Examples
    --------
    >>> from aeon.transformations.collection import MinMaxScaler
    >>> import numpy as np
    >>> X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    >>> minmax = MinMaxScaler()
    >>> Xt = minmax.fit_transform(X)
    >>> min_val = np.min(Xt, axis=-1)
    >>> max_val = np.max(Xt, axis=-1)
    >>> assert np.allclose(min_val, 0)
    >>> assert np.allclose(max_val, 1)

    """

    _tags = {
        "X_inner_type": ["numpy3D", "np-list"],
        "fit_is_empty": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
    }

    def __init__(self, min: float = 0, max: float = 1):
        self.min = min
        self.max = max
        super().__init__()

    def _transform(self, X, y=None) -> np.ndarray:
        """
        Transform method to apply the MinMaxScaler normalisation.

        Parameters
        ----------
        X: np.ndarray or list
            Collection to transform. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        y: None
           Ignored.

        Returns
        -------
        np.ndarray or list
            The data transformed onto [min,max] scale.
        """
        if self.min > self.max:
            raise ValueError(
                f"min value {self.min} should be less than max value {self.max}"
            )

        if isinstance(X, np.ndarray):
            # Case 1: Equal length series, X is a single 3D array
            min_val = np.min(X, axis=-1, keepdims=True)
            max_val = np.max(X, axis=-1, keepdims=True)
            range_val = np.where(
                max_val - min_val == 0, 1, max_val - min_val
            )  # Prevent division by zero
            X_scaled = (X - min_val) / range_val
            X_scaled = self.min + X_scaled * (self.max - self.min)
            return X_scaled
        else:
            # Case 2: Unequal length series, X is a list of 2D arrays
            Xt = []
            for x in X:
                min_val = np.min(x, axis=-1, keepdims=True)
                max_val = np.max(x, axis=-1, keepdims=True)
                range_val = np.where(
                    max_val - min_val == 0, 1, max_val - min_val
                )  # Prevent division by zero
                x_scaled = (x - min_val) / range_val
                x_scaled = self.min + x_scaled * (self.max - self.min)
                Xt.append(x_scaled)
            return Xt


class Centerer(BaseCollectionTransformer):
    """Centering transformer for collections.

    This transformer recentres series to have zero mean, but does not change the
    variance. For multivariate data, it normalizes each channel independently.

    Examples
    --------
    >>> from aeon.transformations.collection import Centerer
    >>> import numpy as np
    >>> X = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    >>> recentre = Centerer()
    >>> Xt = recentre.fit_transform(X)
    >>> mean=np.mean(Xt, axis=-1)
    >>> assert np.allclose(mean, 0)
    """

    _tags = {
        "X_inner_type": ["numpy3D", "np-list"],
        "fit_is_empty": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
    }

    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None) -> np.ndarray:
        """
        Transform method to center series.

        Parameters
        ----------
        X : np.ndarray or list
            Collection to transform. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        y : None
            Ignored.

        Returns
        -------
        np.ndarray or list
            The centered data.
        """
        if isinstance(X, np.ndarray):
            # Case 1: X is a single 3D array
            mean_val = np.mean(X, axis=-1, keepdims=True)
            Xt = X - mean_val
            return Xt

        else:
            # Case 2: X is a list of 2D arrays
            Xt = []
            for x in X:
                mean_val = np.mean(x, axis=-1, keepdims=True)
                xt = x - mean_val
                Xt.append(xt)
            return Xt
