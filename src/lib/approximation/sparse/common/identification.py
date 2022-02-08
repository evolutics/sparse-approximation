import numpy


def shift(*, C, p, D, q):
    residual = p - q
    normalized_residual = residual / _length(D=D, Q=residual)

    c_shift = C - q[:, None]
    normalized_c_shift = c_shift / _length(D=D, Q=c_shift)

    return D(normalized_residual, normalized_c_shift)


def _length(*, D, Q):
    m = Q.shape[0]
    return D(numpy.zeros(m), Q)
