from pytest import mark
import numpy

from src.lib import sorting


@mark.parametrize(
    "values,count,expected_indices",
    [
        (numpy.array([]), 0, numpy.array([])),
        (numpy.array([100]), 0, numpy.array([])),
        (numpy.array([100]), 1, numpy.array([0])),
        (numpy.array([100, 101]), 1, numpy.array([1])),
        (numpy.array([101, 100]), 1, numpy.array([0])),
        (numpy.array([100, 100]), 2, numpy.array([0, 1])),
        (numpy.array([101, 103, 100, 102]), 2, numpy.array([1, 3])),
        (numpy.array([101, 100, 100, 101]), 2, numpy.array([0, 3])),
        (numpy.array([100, 101, 102, 100, 102]), 3, numpy.array([1, 2, 4])),
    ],
)
def test_argmaxs(values, count, expected_indices):
    actual_indices = sorting.argmaxs(values, count)

    assert all(numpy.sort(actual_indices) == expected_indices)


@mark.parametrize(
    "values,count,expected_indices",
    [
        (numpy.array([]), 0, numpy.array([])),
        (numpy.array([100]), 0, numpy.array([])),
        (numpy.array([100]), 1, numpy.array([0])),
        (numpy.array([100, 101]), 1, numpy.array([0])),
        (numpy.array([101, 100]), 1, numpy.array([1])),
        (numpy.array([100, 100]), 2, numpy.array([0, 1])),
        (numpy.array([101, 103, 100, 102]), 2, numpy.array([0, 2])),
        (numpy.array([101, 100, 100, 101]), 2, numpy.array([1, 2])),
        (numpy.array([101, 100, 102, 101, 102]), 3, numpy.array([0, 1, 3])),
    ],
)
def test_argmins(values, count, expected_indices):
    actual_indices = sorting.argmins(values, count)

    assert all(numpy.sort(actual_indices) == expected_indices)
