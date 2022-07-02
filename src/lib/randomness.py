import numpy


def draw_distribution(generator, size, nonzero_range=None):
    if nonzero_range is None:
        nonzero_count = size
    else:
        nonzero_count = generator.integers(*nonzero_range)

    indices = generator.choice(size, nonzero_count, replace=False)

    cumulative_distribution = numpy.sort(generator.random(nonzero_count - 1))
    probabilities = numpy.diff(cumulative_distribution, prepend=0, append=1)

    distribution = numpy.zeros(size)
    distribution[indices] = probabilities

    return distribution


def draw_distributions(generator, rows, columns, nonzero_range=None):
    return numpy.column_stack(
        [
            draw_distribution(generator, rows, nonzero_range=nonzero_range)
            for _ in range(columns)
        ]
    )


def draw_nonnegative_matrix(generator, rows, columns, nonzero_range=None):
    return numpy.column_stack(
        [
            draw_nonnegative_vector(generator, rows, nonzero_range=nonzero_range)
            for _ in range(columns)
        ]
    )


def draw_nonnegative_vector(generator, size, nonzero_range=None):
    if nonzero_range is None:
        nonzero_count = size
    else:
        nonzero_count = generator.integers(*nonzero_range)
    indices = generator.choice(size, nonzero_count, replace=False)

    if nonzero_range is None:
        expected_nonzero_count = size
    else:
        expected_nonzero_count = (sum(nonzero_range) - 1) / 2
    expected_weight = expected_nonzero_count * 0.5
    nonzero_values = generator.random(nonzero_count) / expected_weight

    result = numpy.zeros(size)
    result[indices] = nonzero_values

    return result
