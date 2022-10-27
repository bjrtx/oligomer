import numpy
import matplotlib.pyplot as plt
import string
import itertools

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA

# You need to be in the directory containing those files.
plot_index = itertools.count(1)
plt.subplots(layout="constrained")

for fname in (
    "OUT_above_TH_by_chain_dom1.csv",
    "OUT_above_TH_by_chain_comb.csv",
    "OUT_sum_by_chain_dom1.csv",
    "OUT_sum_by_chain_comb.csv",
):
    data = numpy.loadtxt(fname, delimiter=",").transpose()
    chain_names = string.ascii_uppercase[:24]  # A to X
    pca = PCA(n_components=2).fit(data)
    reduced = pca.transform(data)

    for method, name in zip((KMeans, SpectralClustering), ("k-means", "spectral")):
        labels = method(n_clusters=2).fit(data).labels_
        # by convention, the first data point will always be in cluster 0
        if labels[0]:
            labels = 1 - labels
        colors = [("navy", "darkorange")[l] for l in labels]
        plt.subplot(3, 3, next(plot_index))
        plt.scatter(reduced[:, 0], reduced[:, 1], color=colors, alpha=0.8)
        plt.title(
            f"""Data: {fname}.
            Clustering: {name}.
            Var. expl. by 2-D PCA: {sum(pca.explained_variance_ratio_):.1%}."""
        )
        for chain, xy in zip(chain_names, reduced):
            plt.annotate(chain, xy)

plt.show()
