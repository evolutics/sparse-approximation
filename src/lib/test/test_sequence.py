from pytest import mark

from src.lib import sequence


@mark.parametrize(
    "start,expected",
    [
        (-2, ()),
        (-1, ()),
        (0, ()),
        (1, (1,)),
        (2, (2, 1)),
        (3, (3, 1)),
        (4, (4, 2, 1)),
        (5, (5, 2, 1)),
        (6, (6, 3, 1)),
        (7, (7, 3, 1)),
        (13, (13, 6, 3, 1)),
    ],
)
def test_halve_until_1(start, expected):
    actual = sequence.halve_until_1(start)

    assert tuple(actual) == expected
