import numpy


def clip(raw):
    clipped = numpy.maximum(raw, 0)
    return clipped / numpy.sum(clipped)
