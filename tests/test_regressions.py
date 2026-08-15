"""Regression tests for correctness and input-handling bugs."""

import numpy as np
import pytest

import significantdigits as sd
from significantdigits._significantdigits import _shuffle_along_axis


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (sd.significant_digits, np.array([3, 20])),
        (sd.contributing_digits, np.array([2, 19])),
    ],
)
def test_general_method_finishes_each_output(metric, expected):
    """A low-precision output must not stop higher-precision outputs early."""
    array = np.array(
        [
            [1.0, 1.0 + 2**-20],
            [1.0 + 2**-3, 1.0 + 2**-20],
            [1.0 - 2**-3, 1.0 - 2**-20],
        ]
    )

    result = metric(
        array,
        reference=np.ones(2),
        axis=0,
        method=sd.Method.General,
    )

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("method_name", "method_enum"),
    [
        ("cnh", sd.Method.CNH),
        ("CNH", sd.Method.CNH),
        ("general", sd.Method.General),
        ("General", sd.Method.General),
    ],
)
@pytest.mark.parametrize("metric", [sd.significant_digits, sd.contributing_digits])
def test_method_names_are_case_insensitive(metric, method_name, method_enum):
    array = np.array([0.99, 1.01, 1.02, 0.98])

    actual = metric(array, reference=1.0, method=method_name)
    expected = metric(array, reference=1.0, method=method_enum)

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("error_name", "error_enum"),
    [
        ("absolute", sd.Error.Absolute),
        ("Absolute", sd.Error.Absolute),
        ("relative", sd.Error.Relative),
        ("Relative", sd.Error.Relative),
    ],
)
@pytest.mark.parametrize("metric", [sd.significant_digits, sd.contributing_digits])
def test_error_names_are_case_insensitive(metric, error_name, error_enum):
    array = np.array([0.99, 1.01, 1.02, 0.98])

    actual = metric(array, reference=1.0, error=error_name)
    expected = metric(array, reference=1.0, error=error_enum)

    np.testing.assert_allclose(actual, expected)


def test_shuffle_along_axis_returns_copy(monkeypatch):
    array = np.arange(12).reshape(3, 4)
    original = array.copy()
    monkeypatch.setattr(np.random, "permutation", lambda size: np.arange(size)[::-1])

    shuffled = _shuffle_along_axis(array, axis=1)

    np.testing.assert_array_equal(shuffled, array[:, ::-1])
    np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize("metric", [sd.significant_digits, sd.contributing_digits])
def test_public_shuffle_does_not_mutate_inputs(metric):
    array = np.arange(1.0, 17.0).reshape(4, 4)
    reference = array + 0.25
    original_array = array.copy()
    original_reference = reference.copy()

    metric(
        array,
        reference=reference,
        axis=0,
        error=sd.Error.Absolute,
        shuffle_samples=True,
    )

    np.testing.assert_array_equal(array, original_array)
    np.testing.assert_array_equal(reference, original_reference)


@pytest.mark.parametrize("metric", [sd.significant_digits, sd.contributing_digits])
def test_public_shuffle_without_reference_does_not_mutate_input(metric):
    array = np.arange(1.0, 17.0).reshape(4, 4)
    original = array.copy()

    metric(
        array,
        axis=0,
        error=sd.Error.Absolute,
        shuffle_samples=True,
    )

    np.testing.assert_array_equal(array, original)
