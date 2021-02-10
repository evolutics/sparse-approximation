import numpy


def argmaxs(values, count):
    """Indices of `count` biggest entries (order undefined)."""

    return numpy.argpartition(values, kth=-count)[-count:]


def argmins(values, count):
    """Indices of `count` smallest entries (order undefined)."""

    return numpy.argpartition(values, kth=count - 1)[:count]
