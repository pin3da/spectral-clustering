import numpy
import scipy.io
import h5py


def load_dot_mat(path, db_name):
    try:
        mat = scipy.io.loadmat(path)
    except NotImplementedError:
        mat = h5py.File(path)

    return numpy.array(mat[db_name]).transpose()
