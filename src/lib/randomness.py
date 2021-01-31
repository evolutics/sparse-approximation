import numpy


def draw_distribution(generator, size, nonzero_count=None):
    if nonzero_count is None:
        nonzero_count = generator.integers(1, size + 1)

    indices = generator.choice(size, nonzero_count, replace=False)

    cumulative_distribution = numpy.sort(generator.random(nonzero_count - 1))
    probabilities = numpy.diff(cumulative_distribution, prepend=0, append=1)

    distribution = numpy.zeros(size)
    distribution[indices] = probabilities

    return distribution


def draw_distributions(generator, rows, columns):
    return numpy.column_stack(
        [draw_distribution(generator, rows) for _ in range(columns)]
    )
