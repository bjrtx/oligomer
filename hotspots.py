from __future__ import annotations  # should avoid some typing problems in Python 3.8

import csv
import operator
import string
import logging
from collections.abc import Iterable

import numpy as np
import skimage
import mrcfile

import learning


def read_mrc(filename: str, dtype=np.float16) -> np.ndarray:
    """
    Read an MRC map from a file and convert it into a Numpy multidimensional array.
    The values are cast to the specified data type, by default 16-bit floats.
    """
    with mrcfile.open(filename) as mrc:
        return mrc.data.astype(dtype)


def gloves(hotspot: dict) -> list[np.ndarray, np.ndarray]:
    """
    Compute the two gloves associated with a given hotspot.
    """
    gloves = [0, 0]

    for i in (1, 2):
        try:
            gloves[i - 1] = read_mrc(hotspot[f"Bfr{i}_file_name"]) > float(
                hotspot[f"Bfr{i}_Molmap_TH"]
            )
        except (FileNotFoundError, ValueError):
            # This happens if the CSV file does not have either bfr info or a threshold,
            # in which case the glove is identically zero.
            logging.warning(f"One glove was empty: {hotspot}.")
    return gloves


def biggest_blob(logical: "np.ndarray[bool]", number: int=1) -> "np.ndarray[bool]":    
    """
    Given a logical multidimensional array, find the number largest connected regions
    and return them as a logical array. By default number is 1 and the result is a single
    array. If number is higher the arrays are added to each other.
    """
    labeled = skimage.measure.label(logical)
    regions = skimage.measure.regionprops(labeled)

    import heapq
    biggest = heapq.nlargest(number, regions, key=lambda r: np.sum(r.image))
    labels = [r.label for r in biggest]
    # Will raise an error if there are fewer than number connected components.
    import functools
    res = functools.reduce(np.logical_or, (labeled == l for l in labels))
    return res

dimer_names = ["aq", "bo", "cv", "du", "ep", "fr", "gk", "hn", "is", "jt", "lw", "mx"]

def process(hotspot_filename: str, map_filename: str, map_threshold: float, by_dimers=False):
    """
    Process a density map and return a 2-dimensional array (rows are hotspots,
    columns are chains, entries are scores). Alternatively, if by_dimers is True
    then the columns are dimers.

    hotspot_filename: CSV file containing the hotspot information, must be saved
        in the UTF8 encoding. It has at least the following named columns:
        - "Bfr1_file_name" and "Bfr2_file_name" contain the paths of MRC files,
        - "Bfr1_Molmap_TH" and "Bfr2_Molmap_TH" contain floating point numbers
          between 0 and 1 intended as thresholds.

    map_filename: MRC file containing an electronic density map.
    map_threshold: this parameter seems unused.
    """

    logging.info(f"Processing {map_filename}. Hotspot information: {hotspot_filename}.")

    with open(hotspot_filename) as hotspot_file:
        hotspots = list(csv.DictReader(hotspot_file))

    TH_chains = 0.42

    if not by_dimers:
        columns = [
            read_mrc(f"Bfr_molmap_chain{char}_res5.mrc") > TH_chains
            for char in string.ascii_uppercase[:24]
        ]
    else:
        columns = [
            read_mrc(f"dimer_plus_heme_{d.upper()}.mrc") > TH_chains
            for d in dimer_names
        ]

    map = read_mrc(map_filename)
    out = np.empty([len(hotspots), len(columns)], dtype=np.float16)

    for i, hotspot in enumerate(hotspots):
        bfr1_glove, bfr2_glove = gloves(hotspot)
        for j, chain_or_dimer in enumerate(columns):
            # To avoid parts of hotspots coming from other chains, only the
            # largest connected component of the hotspot-chain intersection
            # is kept.
            bfr1_spot = biggest_blob(bfr1_glove & chain_or_dimer, number=2 if by_dimers else 1)
            bfr2_spot = biggest_blob(bfr2_glove & chain_or_dimer, number=2 if by_dimers else 1)
            comb_size = np.sum(bfr1_spot ^ bfr2_spot)
            # The disjoint union is empty if the two domains are equal - very unlikely.
            if not comb_size:
                output = .0
                logging.warning("A residue has two identical gloves.")
            else:
                output = (
                    map.sum(where=bfr1_spot) - map.sum(where=bfr2_spot)
                ) / comb_size
            logging.info(
                f"hotspot {i + 1} chain "
                f"{(dimer_names if by_dimers else string.ascii_uppercase)[j]} "
                f"comb. value {output:.4}"
            )
            out[i, j] = output
    return out


if __name__ == "__main__":
    # There must be a better way.
    map_filename = "284postprocess.mrc"
    map_threshold = 0.04
    hotspot_filename = "bbRefinedHotSpotsListDaniel.csv"
    out = process(hotspot_filename, map_filename, map_threshold, by_dimers=True)
    learning.analyze(out.transpose())
