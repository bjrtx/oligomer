import csv
import operator
import string
import logging
from collections.abc import Iterable
from __future__ import annotations  # should avoid some typing problems in Python 3.8

import numpy as np
import skimage
import mrcfile

import learning


def read_mrc(filename: str) -> np.ndarray:
    """
    Read an MRC map and return a Numpy multidimensional array.
    """
    with mrcfile.open(filename) as mrc:
        return mrc.data.astype(np.float16)  # rounding to half floats


def gloves(hotspot: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the two gloves associated with a given hotspot.
    """
    try:
        bfr1_glove = read_mrc(hotspot["Bfr1_file_name"]) > float(
            hotspot["Bfr1_Molmap_TH"]
        )
    except (FileNotFoundError, ValueError):
        # This happens if the CSV file does not have either bfr info or a threshold
        bfr1_glove = 0
        logging.warning(f"One glove was empty: {hotspot}.")
    try:
        bfr2_glove = read_mrc(hotspot["Bfr2_file_name"]) > float(
            hotspot["Bfr2_Molmap_TH"]
        )
    except (FileNotFoundError, ValueError):
        bfr2_glove = 0
        logging.warning(f"One glove was empty: {hotspot}.")
    return bfr1_glove, bfr2_glove


def biggest_blob(logical: "np.ndarray[bool]") -> "np.ndarray[bool]":
    """
    Given a logical multidimensional array, find the largest connected region and return it as a logical array.
    """
    # labeled is an ndarray of the same size with each pixel labeled by its connected component
    labeled = skimage.measure.label(logical)
    regions = skimage.measure.regionprops(labeled)
    try:
        biggest = max(regions, key=lambda r: np.sum(r.image)).label
    except ValueError:  # empty list
        return 0
        logging.warning("A chain does not intersect a hotspot.")
    return logical == biggest


def process(hotspot_filename: str, map_filename: str, map_threshold: float):
    """
    Process a density map and return a 2-dimensional array (rows are hotspots,
    columns are chains, entries are scores).

    hotspot_filename: CSV file containing the hotspot information, must be saved
        in the UTF8 encoding.
    map_filename: MRC file containing an electronic density map.
    map_threshold: this parameter seems unused.
    """

    logging.info(f"Processing {map_filename}. Hotspot information: {hotspot_filename}.")

    with open(hotspot_filename) as hotspot_file:
        hotspots = list(csv.DictReader(hotspot_file))

    TH_chains = 0.42
    chains = [
        read_mrc(f"Bfr_molmap_chain{char}_res5.mrc") > TH_chains
        for char in string.ascii_uppercase[:24]
    ]

    map = read_mrc(map_filename)
    out = np.empty([len(hotspots), len(chains)], dtype=np.float16)

    for i, hotspot in enumerate(hotspots):
        bfr1_glove, bfr2_glove = gloves(hotspot)
        for j, chain in enumerate(chains):
            # To avoid parts of hotspots coming from other chains, only the
            # largest connected component of the hotspot-chain intersection
            # is kept.
            bfr1_spot = biggest_blob(bfr1_glove & chain)
            bfr2_spot = biggest_blob(bfr2_glove & chain)
            comb_size = np.sum(bfr1_spot ^ bfr2_spot)
            if not comb_size:
                output = 0
                logging.warning("A residue has two identical gloves.")
            else:
                output = (
                    map.sum(where=bfr1_spot) - map.sum(where=bfr2_spot)
                ) / comb_size
            logging.info(
                f"hotspot {i + 1} chain {string.ascii_uppercase[j]} comb. value {output:.4}"
            )
            out[i, j] = output

    return out


if __name__ == "__main__":
    # There must be a better way.
    map_filename = "284postprocess.mrc"
    map_threshold = 0.04
    hotspot_filename = "bbRefinedHotSpotsListDaniel.csv"
    out = process(hotspot_filename, map_filename, map_threshold)

    learning.analyze(out.transpose())
