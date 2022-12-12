import logging
import string

import numpy
import seaborn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import hotspots


def analyze(data: numpy.ndarray, group_data: bool = True):
    """
    Read a data file whose rows correspond to chains and whose columns correspond to
    hotspots. Conduct statistical analysis.
    The analysis is conducted by dimer when group_data is True and otherwise by chain.
    """
    assert data.shape[0] in (12, 24)  # dimers or chains
    by_dimers = data.shape[0] == 12  # whether the rows are in fact dimers
    if not by_dimers and group_data:
        # if the data is given by chain but we want to analyze it by dimer, we need to
        # group said chains into dimers
        shuffler = [
            (ord(x) - ord("a"), ord(y) - ord("a")) for x, y in hotspots.dimer_names
        ]
        first, second = zip(*shuffler)
        data = numpy.hstack((data[first, :], data[second, :]))
        by_dimers = True
    chain_names = hotspots.dimer_names if by_dimers else string.ascii_uppercase[:24]

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
    color_range = ("darkorange", "navy")
    labels = KMeans(n_clusters=2).fit(data).labels_
    # By convention, the first data point will always be in cluster 0
    labels ^= labels[0]
    colors = [color_range[label] for label in labels]
    # Plot the first PCA component
    seaborn.swarmplot(x=reduced[:, 0], y=chain_names)
    plt.title(
        f"""
        First component in the Principal Component Analysis of the hotspot data.
        Var. expl. by 1-D PCA: {pca.explained_variance_ratio_[0]:.1%}.
        """,
        loc="left",
    )
    plt.ylabel("Dimer name" if by_dimers else "Chain name")
    plt.show()
    # Plot the first two PCA components
    seaborn.scatterplot(x=reduced[:, 0], y=reduced[:, 1], color=colors)
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    for chain, xy in zip(chain_names, reduced):
        plt.annotate(chain, xy, textcoords="offset points", xytext=(3, 3))
    plt.title(
        f"""
        Principal Component Analysis of the hotspot data.
        Var. expl. by 1-D PCA: {pca.explained_variance_ratio_[0]:.1%}.
        Var. expl. by 2-D PCA: {sum(pca.explained_variance_ratio_):.1%}.
        """,
        loc="left",
    )
    logging.info(f"First two components {pca.components_[:2]}")
    plt.show()
