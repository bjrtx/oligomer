import csv
import operator
import string
import numpy as np
import skimage
import mrcfile


def read_mrc(filename) -> np.ndarray:
    with mrcfile.open(filename) as mrc:
        return mrc.data

def gloves(hotspot):
    try:
        bfr1_glove = read_mrc(hotspot['Bfr1_file_name']) > float(hotspot['Bfr1_Molmap_TH'])
    except (FileNotFoundError, ValueError):
        # This happens if the CSV file does not have either bfr info or a threshold
        bfr1_glove = 0
    try:
        bfr2_glove = read_mrc(hotspot['Bfr2_file_name']) > float(hotspot['Bfr2_Molmap_TH'])
    except (FileNotFoundError, ValueError):
        bfr2_glove = 0
    return bfr1_glove, bfr2_glove
    

def biggest_blob(logical : 'np.ndarray[bool]'):
    # labeled is an ndarray of the same size with each pixel labeled by its connected component
    labeled = skimage.measure.label(logical)
    regions = skimage.measure.regionprops(labeled)
    try:
        biggest = max(
            regions,
            key=lambda r: np.sum(r.image)
        ).label
    except ValueError: # empty list
        return 0
    return logical == biggest
    # raise an error if max of empty list


def process(hotspot_filename, map_filenames, map_thresholds):

    with open(hotspot_filename) as hotspot_file:
        hotspots = list(csv.DictReader(hotspot_file))

    chains = (
        read_mrc(f"Bfr_molmap_chain{char}_res5.mrc")
        for char in string.ascii_uppercase[:24]
    )

    TH_chains = 0.42
    logical_chains = [chain > TH_chains for chain in chains]

    for filename, threshold in zip(map_filenames, map_thresholds):
        map = read_mrc(filename)
        for i, hotspot in enumerate(hotspots):
            bfr1_glove, bfr2_glove = gloves(hotspot)
            for j, chain in enumerate(logical_chains):
                # biggest_blob is used to avoid parts of hotspots coming from other chains
                bfr1_spot = biggest_blob(bfr1_glove & chain)
                bfr2_spot = biggest_blob(bfr2_glove & chain)
                comb_size = np.sum(bfr1_spot ^ bfr2_spot)
                output = (np.sum(map[bfr1_spot]) - np.sum(map[bfr2_spot])) / comb_size
                print(f"hotspot {i + 1} chain {string.ascii_uppercase[j]} comb. value {output}")

if __name__ == "__main__":
    # There must be a better way.
    map_filenames = ["284postprocess.mrc"]
    map_thresholds = [0.04]
    hotspot_filename = "RefinedHotSpotsListDaniel.csv"
    process(hotspot_filename, map_filenames, map_thresholds)