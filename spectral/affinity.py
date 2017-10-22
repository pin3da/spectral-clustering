from functools import partial

import numpy
from scipy.optimize import minimize

from .kernels import squared_exponential, radial_basis_phi


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


def _log_sq(x, eps=1e-14):
    t = numpy.log(x + eps)
    return t * t


def _auto_prunning_cost(X, K, b, v, gamma=0.5):
    kernel = partial(radial_basis_phi, b=b, v=v)
    K_bv = compute_affinity(X, kernel)
    num = numpy.linalg.norm(K_bv - K)
    den = numpy.sqrt(numpy.linalg.norm(K_bv) * numpy.linalg.norm(K))
    rho = _log_sq(num / den)
    n = X.shape[0]
    s = 1.0 - (numpy.count_nonzero(K_bv) / (n * n))
    s = _log_sq(s)
    return numpy.sqrt((1.0 - gamma) * rho + gamma * s)


def _auto_prunning_find_b(X, v, affinity):
    K = affinity(X)

    def cost_b(x):
        return -1 * _auto_prunning_cost(X, K, x, v)  # we need to maximize this function

    result = minimize(
        cost_b,
        [numpy.mean(K)],
        bounds=((0, None),),  # positive
        # options={'disp': True, 'maxiter': 100})
    )
    print('best b ', result.x[0])
    return result.x[0]


def automatic_prunning(X, affinity=com_aff_local_scaling):
    D = X.shape[1]
    v = (D + 1) / 2
    b = _auto_prunning_find_b(X, v, affinity)

    affinity = affinity(X)
    N = X.shape[0]
    ans = numpy.zeros((N, N))
    for i in range(N):
        for j in range(N):
            ans[i][j] = radial_basis_phi(X[i], X[j], b, v) * affinity[i][j]
    return ans
