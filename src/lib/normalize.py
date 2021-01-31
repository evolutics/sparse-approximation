import numpy


def clip(raw):
    clipped = numpy.maximum(raw, 0)
    return clipped / numpy.sum(clipped)


def shift(raw):
    shifted = raw - min(numpy.min(raw), 0)
    return shifted / numpy.sum(shifted)
