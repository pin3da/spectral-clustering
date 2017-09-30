import matplotlib.pyplot as plt
import seaborn

from spectral import utils, affinity


seaborn.set()
X = utils.load_dot_mat('data/DB.mat', 'DB/happy')
plt.plot(X[:,0], X[:,1], 'ro')
plt.show()

affinity = affinity.compute_affinity(X)
seaborn.heatmap(affinity)
plt.show()
