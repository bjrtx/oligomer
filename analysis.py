import itertools
import math
import logging

import numpy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
#from sklearn.mixture import GaussianMixture

import hotspots


def analyze(data: numpy.ndarray, group_data: bool = True):
    """
    Read a data file whose rows correspond to chains and whose columns correspond to
    hotspots. Conduct statistical analysis.
    The analysis is conducted by dimer when group_data is True and otherwise by chain.
    """
    by_dimers = data.shape[0] == 12  # the rows are in fact dimers
    plot_index = itertools.count(1)
    if not by_dimers and group_data:
        shuffler = [
            (ord(x) - ord("a"), ord(y) - ord("a")) for x, y in hotspots.dimer_names
        ]
        first, second = zip(*shuffler)
        data = numpy.hstack((data[first, :], data[second, :]))
        by_dimers = True
    chain_names = hotspots.dimer_names if by_dimers else string.ascii_uppercase[:24]

    pca = PCA(n_components=2, whiten=True).fit(data)
    reduced = pca.transform(data)
    color_range = ("darkorange", "navy")
    labels = KMeans(n_clusters=2).fit(data).labels_
    # By convention, the first data point will always be in cluster 0
    labels = labels != labels[0]
    colors = [color_range[l] for l in labels]
    plt.scatter(reduced[:, 0], reduced[:, 1], color=colors, alpha=0.8)
    for chain, xy in zip(chain_names, reduced):
        plt.annotate(chain, xy, textcoords="offset points", xytext=(3, 3))
    plt.title(
        f"""
        Data: sum by chain.
        Clustering: k-means.
        Var. expl. by 1-D PCA: {pca.explained_variance_ratio_[0]:.1%}.
        Var. expl. by 2-D PCA: {sum(pca.explained_variance_ratio_):.1%}.
        """,
        loc="left",
    )
    logging.info(f"First two components {pca.components_[:2]}")
    # for n_components in (2,):
    #     gm = GaussianMixture(
    #         n_components, n_init=20, max_iter=2000, covariance_type="spherical"
    #     ).fit(data)
    #     labels = gm.predict(data)
    #     colors = [color_range[l] for l in labels]
    #     plt.subplot(dim, dim, next(plot_index))
    #     plt.scatter(reduced[:, 0], reduced[:, 1], color=colors, alpha=0.8)
    #     for chain, xy in zip(chain_names, reduced):
    #         plt.annotate(chain, xy, textcoords="offset points", xytext=(3, 3))
    #     plt.title(
    #         # Data: {filename.split(".")[0]}.
    #         f"""
    #         Data: sum by chain.
    #         Gaussian mixture: {n_components} components.
    #         Var. expl. by 1-D PCA: {pca.explained_variance_ratio_[0]:.1%}.
    #         Var. expl. by 2-D PCA: {sum(pca.explained_variance_ratio_):.1%}.
    #         """,
    #         loc="left",
    #     )
    plt.show()
