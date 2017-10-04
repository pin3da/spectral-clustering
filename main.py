from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy
import seaborn

from spectral import utils, affinity, clustering

seaborn.set()

# ALL datasets: "be2" "be3" "ch" "dc3" "ds3" "ds4" "ds5" "g4" "happy" "hm" "hm2" "iris" "rollo2" "sp" "tar"
# working examples:
#   be2: sig 0.1
#   ds4: sig 0.01
#   happy: sig 0.03
#   hm: sig 0.1
#   sp: sig 1
#   tar: sig 0.5
X = utils.load_dot_mat('data/DB.mat', 'DB/tar')
N = X.shape[0]
K = 6

print("Performing spectral clustering on %d items and %d clases" % (N, K))
A = affinity.compute_affinity(X)

Y = clustering.spectral_clustering(A, K)
Y2 = clustering.k_means(X, K)

colors = numpy.array(list(islice(cycle(seaborn.color_palette()), int(max(Y) + 1))))

plt.scatter(X[:, 0], X[:, 1], color=colors[Y])
plt.show()
plt.scatter(X[:, 0], X[:, 1], color=colors[Y2])
plt.show()
