import logging
import string

import numpy
import seaborn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sklearn.preprocessing

import hotspots


def analyze(data: numpy.ndarray, group_data: bool = True,
            all_bfr1: numpy.ndarray | None = None,
            all_bfr2: numpy.ndarray | None = None):
    """
    Read a data file whose rows correspond to chains and whose columns correspond to
    hotspots. Conduct statistical analysis.
    The analysis is conducted by dimer when group_data is True and otherwise by chain.
    If both all_bfr1 and all_bfr2 are passed, they should contain hotspot data for an
    all-bfr1 and an all-bfr2 map. In this case, two data points for all_bfr1 and
    all_bfr2 are added.
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
    # The dimer names are found in hotspots. The chains alone have names from A to X.
    chain_names = hotspots.dimer_names if by_dimers else string.ascii_uppercase[:24]

    all_bfr = (all_bfr1 is not None) and (all_bfr2 is not None) and by_dimers
    # Do we want to add comparison points for all-Bfr1 and all-Bfr2 structures?
    if all_bfr:
        logging.log("Received additiona data for homogeneous structures.")
        # generated all_bfr data is not to the same scale as cryo-EM data
        # thus we need to center and scale each dimer (losing two dimensions in the
        # process)
        #sklearn.preprocessing.scale(data, axis=1, copy=False)
        #sklearn.preprocessing.scale(all_bfr1, axis=1, copy=False)
        #sklearn.preprocessing.scale(all_bfr2, axis=1, copy=False)
        print(all_bfr1.mean(axis=1), all_bfr2.mean(axis=1))
        data = numpy.vstack((data, all_bfr1, all_bfr2))
        print("augmented data", data)
        chain_names += ["all_bfr1", "all_bfr2"]
        # data.astype(numpy.float64, copy=False)
    pca = PCA(n_components=2)
    if all_bfr:
        reduced = pca.fit(data[:-2, :]).transform(data)
    else:
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
