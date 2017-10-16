import numpy

from .kernels import squared_exponential


def compute_affinity(X, kernel=squared_exponential):
    N = X.shape[0]
    ans = numpy.zeros((N, N))
    for i in range(N):
        for j in range(N):
            ans[i][j] = kernel(X[i], X[j])
    return ans


def com_aff_local_scaling(X):
    N = X.shape[0]
    ans = numpy.zeros((N, N))
    sig = []
    for i in range(N):
        dists = []
        for j in range(N):
            dists.append(numpy.linalg.norm(X[i] - X[j]))
        dists.sort()
        sig.append(numpy.mean(dists[:5]))

    for i in range(N):
        for j in range(N):
            ans[i][j] = squared_exponential(X[i], X[j], sig[i], sig[j])
    return ans
