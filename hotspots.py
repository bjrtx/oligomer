import csv
import operator
import string
from collections.abc import Iterable
import numpy as np
import skimage
import mrcfile


def read_mrc(filename: str) -> np.ndarray:
    """
    Read an MRC map and return a Numpy multidimensional array.
    """
    with mrcfile.open(filename) as mrc:
        return mrc.data


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
    try:
        bfr2_glove = read_mrc(hotspot["Bfr2_file_name"]) > float(
            hotspot["Bfr2_Molmap_TH"]
        )
    except (FileNotFoundError, ValueError):
        bfr2_glove = 0
    return bfr1_glove, bfr2_glove

def biggest_blob(logical : 'np.ndarray[bool]') -> 'np.ndarray[bool]':
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
    return logical == biggest
    # raise an error if max of empty list


def process(
    hotspot_filename: str, map_filenames: Iterable[str], map_thresholds: Iterable[float]
):

    with open(hotspot_filename) as hotspot_file:
        hotspots = list(csv.DictReader(hotspot_file))

    TH_chains = 0.42
    chains = [
        read_mrc(f"Bfr_molmap_chain{char}_res5.mrc") > TH_chains
        for char in string.ascii_uppercase[:24]
    ]

    for filename, threshold in zip(map_filenames, map_thresholds):
        map = read_mrc(filename)
        out = np.empty([len(hotspots), len(chains)])
        for i, hotspot in enumerate(hotspots):
            bfr1_glove, bfr2_glove = gloves(hotspot)
            for j, chain in enumerate(chains):
                # biggest_blob is used to avoid parts of hotspots coming from other chains
                bfr1_spot = biggest_blob(bfr1_glove & chain)
                bfr2_spot = biggest_blob(bfr2_glove & chain)
                comb_size = (bfr1_spot ^ bfr2_spot).sum()
                output = (map.sum(where=bfr1_spot) - map.sum(where=bfr2_spot)) / comb_size
                print(f"hotspot {i + 1} chain {string.ascii_uppercase[j]} comb. value {output}")
                out[i, j] = output

    return out


if __name__ == "__main__":
    # There must be a better way.
    map_filenames = ["284postprocess.mrc"]
    map_thresholds = [0.04]
    hotspot_filename = "RefinedHotSpotsListDaniel.csv"
    process(hotspot_filename, map_filenames, map_thresholds)
