from itertools import cycle, islice
from functools import partial

import matplotlib.pyplot as plt
import numpy
import seaborn

from spectral import utils, affinity, clustering

seaborn.set()

methods = [
    (affinity.compute_affinity, 'Basic Affinity'),
    (affinity.com_aff_local_scaling, 'Affinity Local Scaling'),
    (affinity.automatic_prunning, 'Auto-pruning + LS'),
    # (partial(affinity.automatic_prunning, affinity=affinity.compute_affinity), 'Auto-pruning'),
]

# ALL datasets: "be2" "be3" "ch" "dc3" "ds3" "ds4" "ds5" "g4" "happy" "hm" "hm2" "iris" "rollo2" "sp" "tar"
data_sets = ['be3', 'happy', 'hm', 'sp', 'tar']

num_classes = {
    'be3': 3,
    'happy': 3,
    'hm': 2,
    'sp': 3,
    'tar': 6,
}

H_plot, W_plot = len(data_sets), len(methods)

plt.figure(figsize=(W_plot * 2 + 3, H_plot * 2 + 3,))
plt.subplots_adjust(left=0.1, right=0.98, bottom=0.05, top=0.98, wspace=0.2,
                    hspace=0.3)

for i in range(W_plot):
    for j in range(H_plot):
        X = utils.load_dot_mat('data/DB.mat', 'DB/' + data_sets[j])
        N = X.shape[0]
        K = num_classes.get(data_sets[j], 3)
        affinity, name = methods[i]
        A = affinity(X)
        print("SC on dataset %r with %d clases with method %r" % (data_sets[j], K, name))
        Y = clustering.spectral_clustering(A, K)

        colors = numpy.array(list(islice(cycle(seaborn.color_palette()), int(max(Y) + 1))))
        plt.subplot(H_plot, W_plot, j * W_plot + i + 1)
        plt.scatter(X[:, 0], X[:, 1], color=colors[Y], s=6, alpha=0.6)
        if j == 0:
            plt.title(name)

# plt.show()
plt.savefig('img/uncentered-auto.png')
