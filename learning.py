import numpy
import matplotlib.pyplot as plt
import string
import itertools
import math

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


# You need to be in the directory containing those files.

dim = 3  # arbitrary
shuffler = [
    (0, 16),
    (1, 14),
    (2, 21),
    (3, 20),
    (4, 15),
    (5, 17),
    (6, 10),
    (7, 13),
    (8, 18),
    (9, 19),
    (11, 22),
    (12, 23),
]


def analyze(data, group_data=True):
    plot_index = itertools.count(1)
    chain_names = string.ascii_uppercase[:24]  # list of letters A to X inclusive
    if group_data:
        first, second = zip(*shuffler)
        data = numpy.hstack((data[first, :], data[second, :]))
        chain_names = [
            chain_names[a] + chain_names[b] for a, b in shuffler
        ]  # list of letters A to X inclusive
    pca = PCA(n_components=2).fit(data)
    reduced = pca.transform(data)
    color_range = ("darkorange", "navy", "green")
    plt.subplots(layout="constrained")
    # for method, name in zip((KMeans, SpectralClustering), ("k-means", "spectral")):
    for method, name in zip((KMeans,), ("k-means",)):
        labels = method(n_clusters=2).fit(data).labels_
        # by convention, the first data point will always be in cluster 0
        labels = labels != labels[0]
        colors = [color_range[l] for l in labels]
        plt.subplot(dim, dim, next(plot_index))
        plt.scatter(reduced[:, 0], reduced[:, 1], color=colors, alpha=0.8)
        for chain, xy in zip(chain_names, reduced):
            plt.annotate(chain, xy, textcoords="offset points", xytext=(3, 3))
        plt.title(
            # Data: {filename.split(".")[0]}.
            f"""
            Data: sum by chain.
            Clustering: {name}.
            Var. expl. by 2-D PCA: {sum(pca.explained_variance_ratio_):.1%}.
            """,
            loc="left",
        )
    for n_components in (2, 3):
        gm = GaussianMixture(
            n_components, n_init=20, max_iter=2000, covariance_type="spherical"
        ).fit(data)
        labels = gm.predict(data)
        colors = [color_range[l] for l in labels]
        plt.subplot(dim, dim, next(plot_index))
        plt.scatter(reduced[:, 0], reduced[:, 1], color=colors, alpha=0.8)
        for chain, xy in zip(chain_names, reduced):
            plt.annotate(chain, xy, textcoords="offset points", xytext=(3, 3))
        plt.title(
            # Data: {filename.split(".")[0]}.
            f"""
            Data: sum by chain.
            Gaussian mixture: {n_components} components.
            Var. expl. by 2-D PCA: {sum(pca.explained_variance_ratio_):.1%}.
            """,
            loc="left",
        )
    plt.show()


if __name__ == "__main__":
    filenames = (
        "OUT_above_TH_by_chain_dom1.csv",
        "OUT_above_TH_by_chain_comb.csv",
        "OUT_sum_by_chain_dom1.csv",
        "OUT_sum_by_chain_comb.csv",
    )
    dim = 1 + math.isqrt(3 * len(filenames) - 1)
    for filename in filenames:
        try:
            data = numpy.loadtxt(filename, delimiter=",").transpose()
        except OSError:
            print("Files not found")
            exit()
        analyze(data)
