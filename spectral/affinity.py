import numpy

def squared_exponential(x, y, sig_sq=0.5):
    norm = numpy.linalg.norm(x - y)
    dist = norm * norm
    return numpy.exp(- dist / 2 * sig_sq)

def compute_affinity(X, kernel=squared_exponential):
    N = X.shape[0]
    ans = numpy.zeros((N, N))
    for i in range(N):
        for j in range(N):
            ans[i][j] = kernel(X[i], X[j])
    return ans
