import numpy
import matplotlib.pyplot as plt
import string
import itertools

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# You need to be in the directory containing those files.
plot_index = itertools.count(1)
plt.subplots(layout="constrained")

for filename in (
    "OUT_above_TH_by_chain_dom1.csv",
    "OUT_above_TH_by_chain_comb.csv",
    "OUT_sum_by_chain_dom1.csv",
    "OUT_sum_by_chain_comb.csv",
):
    data = numpy.loadtxt(filename, delimiter=",").transpose()
    chain_names = string.ascii_uppercase[:24]  # list of letters A to X inclusive
    pca = PCA(n_components=2).fit(data)
    reduced = pca.transform(data)
    # for method, name in zip((KMeans, SpectralClustering), ("k-means", "spectral")):
    for method, name in zip((KMeans,), ("k-means",)):
        labels = method(n_clusters=2).fit(data).labels_
        # by convention, the first data point will always be in cluster 0
        labels = labels != labels[0]
        colors = [("darkorange", "navy")[l] for l in labels]
        plt.subplot(3, 3, next(plot_index))
        plt.scatter(reduced[:, 0], reduced[:, 1], color=colors, alpha=0.8)
        for chain, xy in zip(chain_names, reduced):
            plt.annotate(chain, xy, textcoords="offset points", xytext=(3, 3))
        plt.title(
            f"""
            Data: {filename.split(".")[0]}.
            Clustering: {name}.
            Var. expl. by 2-D PCA: {sum(pca.explained_variance_ratio_):.1%}.
            """,
            loc="left",
        )
    for n_components in (2, 3):
        gm = GaussianMixture(n_components, n_init=20, max_iter=2000).fit(data)
        labels = gm.predict(data)
        colors = [("darkorange", "navy", "green")[l] for l in labels]
        plt.subplot(3, 3, next(plot_index))
        plt.scatter(reduced[:, 0], reduced[:, 1], color=colors, alpha=0.8)
        for chain, xy in zip(chain_names, reduced):
            plt.annotate(chain, xy, textcoords="offset points", xytext=(3, 3))
        plt.title(
            f"""
            Data: {filename.split(".")[0]}.
            Gaussian mixture: {n_components} components.
            Var. expl. by 2-D PCA: {sum(pca.explained_variance_ratio_):.1%}.
            """,
            loc="left",
        )


plt.show()
