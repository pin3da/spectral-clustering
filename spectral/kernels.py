import numpy


def radial_basis_phi(x, y, b, v):
    norm = numpy.linalg.norm(x - y)
    return max(1.0 - norm / b, 0) ** v


def squared_exponential(x, y, sig=0.8, sig2=1):
    norm = numpy.linalg.norm(x - y)
    dist = norm * norm
    return numpy.exp(- dist / (2 * sig * sig2))
