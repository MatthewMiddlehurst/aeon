"""Tests for ElasticEnsemble."""

import numpy as np
import pytest
from sklearn.model_selection import StratifiedShuffleSplit

from aeon.classification.distance_based import ElasticEnsemble

DISTANCE = [
    "dtw",
    "ddtw",
    "wdtw",
    "wddtw",
    "lcss",
    "erp",
    "msm",
    "euclidean",
    "twe",
]

PARAS = {
    "dtw": {"window"},
    "ddtw": {"window"},
    "wdtw": {"g"},
    "wddtw": {"g"},
    "lcss": {"epsilon", "window"},
    "erp": {"g", "window"},
    "msm": {"c"},
    "euclidean": {},
    "twe": {"nu", "lmbda", "window"},
}
DATA = np.random.random((10, 1, 50))


@pytest.mark.parametrize("dist", DISTANCE)
@pytest.mark.parametrize("data", [None, DATA])
def test_get_100_param_options(dist, data):
    """Test the method to get 100 options per distance function.

    1. Test 100 returned.
    2. Test on specified range.
    """
    if (
        dist == "erp" or dist == "lcss"
    ) and data is None:  # raise exception, LCSS needs
        # train data
        with pytest.raises(ValueError):
            ElasticEnsemble._get_100_param_options(dist, data)
    else:
        paras = ElasticEnsemble._get_100_param_options(dist, data)
        para_values = paras["distance_params"]
        # Check correct number of para combos
        if dist == "euclidean":
            assert len(para_values) == 1
        else:
            assert len(para_values) == 100
        # Check all provided parameters are valid for this distance
        expected_paras = PARAS[dist]
        for p in para_values:
            for ep in expected_paras:
                assert ep in p


def test_proportion_train_in_param_finding():
    """Test proportion in train for parameter finding."""
    X = np.random.random(size=(10, 1, 10))
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # floor(0.1 * 10) < n_classes, so StratifiedShuffleSplit cannot split.
    ee = ElasticEnsemble(
        distance_measures=["dtw"], proportion_train_in_param_finding=0.1
    )
    ee.fit(X, y)
    assert len(ee.predict(X)) == len(y)
    ee = ElasticEnsemble(
        distance_measures=["ddtw", "wddtw"],
        proportion_train_in_param_finding=0.4,
        verbose=True,
    )
    ee.fit(X, y)
    ee.get_metric_params()
    with pytest.raises(NotImplementedError, match="EE does not currently support:"):
        ElasticEnsemble._get_100_param_options("FOO")


def test_proportion_train_in_param_finding_many_classes():
    """Test subsampling when floor(proportion * n_cases) < n_classes."""
    X = np.random.random(size=(30, 1, 10))
    y = np.repeat(np.arange(10), 3)
    ee = ElasticEnsemble(
        distance_measures=["dtw"], proportion_train_in_param_finding=0.1
    )
    train_index = ee._param_finding_train_index(X, y, np.random.RandomState(0))
    assert np.all(np.bincount(y[train_index]) == 2)
    ee.fit(X, y)
    assert len(ee.predict(X)) == len(y)


def test_proportion_train_in_param_finding_small_minority_class():
    """Test a minority class StratifiedShuffleSplit samples fewer than 2 cases of."""
    X = np.random.random(size=(20, 1, 10))
    y = np.array([0] * 16 + [1] * 4)
    ee = ElasticEnsemble(
        distance_measures=["dtw", "ddtw"], proportion_train_in_param_finding=0.1
    )
    train_index = ee._param_finding_train_index(X, y, np.random.RandomState(0))
    assert len(np.unique(train_index)) == len(train_index)
    assert np.sum(y[train_index] == 1) == 2
    ee.fit(X, y)
    assert len(ee.predict(X)) == len(y)


def test_proportion_train_in_param_finding_singleton_class():
    """Test a class with a single case is kept, with a warning."""
    X = np.random.random(size=(11, 1, 10))
    y = np.array([0] * 5 + [1] * 5 + [2])
    ee = ElasticEnsemble(
        distance_measures=["dtw"], proportion_train_in_param_finding=0.2
    )
    train_index = ee._param_finding_train_index(X, y, np.random.RandomState(0))
    assert list(np.bincount(y[train_index])) == [2, 2, 1]
    with pytest.warns(UserWarning, match="less than 2 cases"):
        ee.fit(X, y)
    assert len(ee.predict(X)) == len(y)


@pytest.mark.parametrize("proportion", [0.0, -0.5, 1.5])
def test_proportion_train_in_param_finding_invalid(proportion):
    """Test proportion_train_in_param_finding outside (0, 1] raises an error."""
    X = np.random.random(size=(10, 1, 10))
    y = np.array([0] * 5 + [1] * 5)
    ee = ElasticEnsemble(
        distance_measures=["dtw"], proportion_train_in_param_finding=proportion
    )
    with pytest.raises(ValueError, match="must be in the range"):
        ee.fit(X, y)


def test_proportion_train_in_param_finding_matches_stratified_split():
    """Test sampling is unchanged when StratifiedShuffleSplit already suffices.

    The random state is passed on to RandomizedSearchCV, so both the sampled
    indices and the random state afterwards must match a plain split.
    """
    X = np.random.random(size=(40, 1, 10))
    y = np.array([0] * 20 + [1] * 20)
    ee = ElasticEnsemble(proportion_train_in_param_finding=0.2)

    rand = np.random.RandomState(1)
    train_index = ee._param_finding_train_index(X, y, rand)

    expected_rand = np.random.RandomState(1)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=expected_rand)
    expected_index, _ = next(sss.split(X, y))

    np.testing.assert_array_equal(train_index, expected_index)
    assert rand.randint(2**31) == expected_rand.randint(2**31)


def test_all_distance_measures():
    """Test the 'all' option of the distance_measures parameter."""
    X = np.random.random(size=(10, 1, 10))
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    ee = ElasticEnsemble(distance_measures="all", proportion_train_in_param_finding=0.4)
    ee.fit(X, y)
    distances = list(ee.get_metric_params())
    assert len(distances) == 11
    for distance in ("dtw_full", "ddtw_full"):
        assert ee.get_metric_params()[distance] == "{'window': 1.0}"


def test_ts_quad_distance_measures():
    """Test the 'ts-quad' option of the distance_measures parameter."""
    X = np.random.random(size=(10, 1, 10))
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    ee = ElasticEnsemble(
        distance_measures="ts-quad",
        proportion_train_in_param_finding=0.4,
        proportion_of_param_options=0.1,
    )
    ee.fit(X, y)
    actual_distances = list(ee.get_metric_params())
    preds = ee.predict(X)
    assert len(actual_distances) == 4
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X)


def test_elastic_ensemble_unequal_length():
    """Test ElasticEnsemble on unequal length time series."""
    rng = np.random.RandomState(0)
    X = [
        rng.normal(0, 1, (1, 5)),
        rng.normal(0, 1, (1, 7)),
        rng.normal(0, 1, (1, 6)),
        rng.normal(0, 1, (1, 8)),
        rng.normal(5, 1, (1, 5)),
        rng.normal(5, 1, (1, 7)),
        rng.normal(5, 1, (1, 6)),
        rng.normal(5, 1, (1, 8)),
    ]
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # Test parameter finding with distance requiring train_std (erp)
    ee = ElasticEnsemble(
        distance_measures=["dtw", "erp"],
        proportion_of_param_options=0.05,
        random_state=0,
    )
    ee.fit(X, y)
    preds = ee.predict(X)
    assert len(preds) == len(y)

    # Test subsampling with unequal lengths
    ee_sub = ElasticEnsemble(
        distance_measures=["dtw"],
        proportion_train_in_param_finding=0.5,
        proportion_of_param_options=0.05,
        random_state=0,
    )
    ee_sub.fit(X, y)
    preds_sub = ee_sub.predict(X)
    assert len(preds_sub) == len(y)
