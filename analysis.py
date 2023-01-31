import logging
import string
from typing import Optional

import numpy
import seaborn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn.preprocessing

import hotspots


def analyze(
    data: numpy.ndarray,
    group_data: bool = True,
    all_bfr1: Optional[numpy.ndarray] = None,
    all_bfr2: Optional[numpy.ndarray] = None,
):
    """
    Read a data file whose rows correspond to chains and whose columns correspond to
    hotspots. Conduct statistical analysis.

    The analysis is conducted by dimer when group_data is True and otherwise by chain.
    If both all_bfr1 and all_bfr2 are passed, they should contain hotspot data for an
    all-bfr1 and an all-bfr2 map. In this case, two dimers representing ideal Bfr1 and
    Bfr2 dimers are added to the data. That is, the data consists of the 12 dimers from
    the empirical data plus two dimers obtained from all_bfr1 and all_bfr2 respectively.
    This is useful for the analysis, when we want to assess how much empirical dimers
    look like simulated Bfr1 or Bfr2 data.
    """
    assert data.shape[0] in (12, 24), "The data must have either 12 or 24 rows."
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
    # The dimer IDs are found in hotspots. The chains alone have IDs from A to X.
    chain_names = hotspots.dimer_names if by_dimers else string.ascii_uppercase[:24]

    # Do we want to add comparison points for all-Bfr1 and all-Bfr2 structures?
    all_bfr = (all_bfr1 is not None) and (all_bfr2 is not None) and by_dimers
    if all_bfr:
        logging.info("Received additional data for homooligomeric structures.")
        # simulated all_bfr data is not to the same scale as cryo-EM data
        # thus we need to scale each dimer (losing dimensions in the
        # process)
        # sklearn.preprocessing.scale with scores="sum" creates numerical errors,
        # hence the choice of min-max scaling
        scale = sklearn.preprocessing.minmax_scale
        scale(data, axis=1, copy=False)
        scale(all_bfr1, axis=1, copy=False)
        scale(all_bfr2, axis=1, copy=False)
        data = numpy.vstack((data, all_bfr1, all_bfr2))
        chain_names += ["all_bfr1", "all_bfr2"]

    # Define the PCA estimator
    pca = PCA(n_components=2)
    if all_bfr:
        # The principal components are learned from the empirical data,
        # not taking into account the all-bfr rows
        reduced = pca.fit(data[:-2, :]).transform(data)
        first_comp = pca.components_[0]
        first_coeffs = {
            chain: numpy.dot(row, first_comp) for (chain, row) in zip(chain_names, data)
        }
        # scale so that the coeffs of simulated all-bfr1 and all-bfr2 are 0 and 1
        first_coeffs = {
            k: (v - first_coeffs["all_bfr1"])
            / (first_coeffs["all_bfr2"] - first_coeffs["all_bfr1"])
            for k, v in first_coeffs.items()
        }
        print("Bfr1/2 proportion", first_coeffs)
    else:
        reduced = pca.fit_transform(data)
    # Plot the first PCA component
    seaborn.swarmplot(x=reduced[:, 0], y=chain_names)
    plt.title(
        f"""
        First component in the Principal Component Analysis of the hotspot data.
        Variance explained by this component: {pca.explained_variance_ratio_[0]:.1%}.
        """,
        loc="left",
    )
    plt.ylabel("Dimer name" if by_dimers else "Chain name")
    plt.show()
    # Plot the first two PCA components
    # seaborn.scatterplot(x=reduced[:, 0], y=reduced[:, 1])
    # plt.xlabel("First principal component")
    # plt.ylabel("Second principal component")
    # for chain, point in zip(chain_names, reduced):
    #     plt.annotate(chain, point, textcoords="offset points", xytext=(3, 3))
    # plt.title(
    #     f"""
    #     Principal Component Analysis of the hotspot data.
    #     Var. expl. by 1-D PCA: {pca.explained_variance_ratio_[0]:.1%}.
    #     Var. expl. by 2-D PCA: {sum(pca.explained_variance_ratio_):.1%}.
    #     """,
    #     loc="left",
    # )
    # logging.info(f"First two components {pca.components_[:2]}")
    # plt.show()
