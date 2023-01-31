from __future__ import annotations  # should avoid some typing problems in Python 3.8

import csv
import operator
import string
import logging
import heapq
import functools
import argparse
from typing import Optional
from collections.abc import Collection

import numpy as np
import skimage
import mrcfile

import analysis


def read_mrc(filename: str, dtype: Optional[type] = None) -> np.ndarray:
    """
    Read an MRC map from a file and convert it into a Numpy multidimensional array.
    The values are cast to the specified data type if one is given.
    """
    with mrcfile.open(filename) as mrc:
        return mrc.data if dtype is None else mrc.data.astype(dtype)


def hotspot_masks(hotspot: dict[str]) -> list[np.ndarray, np.ndarray]:
    """
    Compute the two masks associated with a given hotspot.

    hotspot: dictionary with the following fields
    - "Bfr1_file_name", the filename for the Bfr1 MRC mask data
    - "Bfr2_file_name", the filename for the Bfr2 MRC mask data
    - "Bfr1_Molmap_TH", the value of the threshold for reading the Bfr1 mask
    - "Bfr2_Molmap_TH", the value of the threshold for reading the Bfr2 mask
    """
    masks = [0, 0]

    for i in (1, 2):
        try:
            masks[i - 1] = read_mrc(hotspot[f"Bfr{i}_file_name"]) > float(
                hotspot[f"Bfr{i}_Molmap_TH"]
            )
        except (FileNotFoundError, ValueError):
            # This happens if the CSV file does not have either Bfr info or a threshold,
            # in which case the mask is identically zero.
            # In practice this should happen only for masks corresponding to hemes.
            logging.warning(
                f"The mask of {hotspot} was missing. This will create more warnings."
            )
    return masks


def biggest_blob(logical: "np.ndarray[bool]", n: int = 1) -> "np.ndarray[bool]":
    """
    Given a logical multidimensional array, find the n largest connected regions
    and return them as a logical array. By default n is 1.
    If n is higher the elementwise maximum of the arrays is returned, which is the
    union of the n largest regions.
    If the input array is all-zero then so is the output.
    """
    # Use image processing methods from scikit-image.
    labeled = skimage.measure.label(logical)
    regions = skimage.measure.regionprops(labeled)
    # regions is a list of connected components.
    # We select its n largest components by area.
    biggest = heapq.nlargest(n, regions, key=operator.attrgetter("area"))
    if len(biggest) < n:
        # This happens if there were fewer than n elements in regions.
        logging.warning(
            f"Fewer connected components than expected ({len(biggest)}/{n}), "
            f"perhaps due to an empty mask."
        )
    # Return the union (logical sum) of all n largest components
    return functools.reduce(np.logical_or, (labeled == r.label for r in biggest), 0)


dimer_names = ["aq", "bo", "cv", "du", "ep", "fr", "gk", "hn", "is", "jt", "lw", "mx"]


def process(
    hotspot_data: str | Collection[dict[str]],
    map_: str | np.ndarray,
    *,  # keyword only arguments
    mask_data: Optional[Collection[tuple[np.ndarray, np.ndarray]]] = None,
    by_dimers: bool = False,
    truncate: bool = True,
    scores: string = "sum",
):
    """
    Process a density map and return a 2-dimensional array (rows are hotspots,
    columns are chains, entries are scores). Alternatively, if by_dimers is True
    then the columns are dimers. Scores can be based on the sum of densities or on
    the number of above-threshold densities.

    hotspot_filename: either a CSV file or a list of rows from a previously read file.
        The CSV file contains hotspot information. It _must_ be saved
        in the UTF8 encoding. It has at least the following named columns:
        - "Bfr1_file_name" and "Bfr2_file_name" contain the paths of MRC files,
        - "Bfr1_Molmap_TH" and "Bfr2_Molmap_TH" contain floating point numbers
          between 0 and 1 intended as thresholds.

    map_: either the path of an MRC file containing an electronic density map, or a map
        that has already been loaded (as a Numpy array).

    truncate: when this is set to True the maps are cast to 16-bit floats, shortening
    computation time. Roughly speaking, this keeps three digits after the decimal point.

    mask_data: either a collection of already computed hotspot masks (3-dimensional
    arrays of Boolean values) or None. If None, the masks will be computed.

    scores: whether to score by sum of electronic density ("sum", the default behaviour)
    or by number of above-threshold values ("threshold")
    """
    if scores not in ("sum", "sum_by_volume", "threshold"):
        raise ValueError('The scores parameter should be "sum", "sum_by_volume" or "threshold".')

    if isinstance(map_, str):
        logging.info(f"Processing new map: {map_}.")
        map_ = read_mrc(map_)

    if truncate:
        map_ = map_.astype(np.float16, casting="same_kind", copy=False)

    if isinstance(hotspot_data, str):
        logging.info(f"Reading hotspot information: {hotspot_data}.")
        with open(hotspot_data, encoding="utf-8") as file:
            hotspot_data = list(csv.DictReader(file))

    chain_threshold = 0.42
    # filenames is the generic form of all dimer / chain MRC map names
    filenames = f"{'dimer' if by_dimers else 'chain'}_plus_heme_?.mrc"
    # indices will keep the dimer names or the chain names
    indices = dimer_names if by_dimers else string.ascii_uppercase[:24]
    # read either dimer or chain MRC maps and convert them into logical arrays
    # by rounding values to 0 or 1 depending on the parameter chain_threshold
    columns = [
        read_mrc(filenames.replace("?", idx)) > chain_threshold for idx in indices
    ]
    result = np.empty([len(hotspot_data), len(columns)], dtype=np.float16)

    if scores == "threshold":
        map_ = (map_ > 0.025).astype(np.float16)  # for future precision printing
        logging.warning("Using threshold scoring.")

    for i, hotspot in enumerate(hotspot_data):
        bfr1, bfr2 = hotspot_masks(hotspot) if mask_data is None else mask_data[i]
        for j, chain_or_dimer in enumerate(columns):
            # To avoid parts of hotspot masks coming from other chains, only the
            # largest connected component of the hotspot-chain intersection
            # is kept.
            n_blobs = 2 if by_dimers else 1
            bfr1_spot = biggest_blob(bfr1 & chain_or_dimer, n_blobs)
            bfr2_spot = biggest_blob(bfr2 & chain_or_dimer, n_blobs)
            assert (
                np.count_nonzero(bfr1_spot ^ bfr2_spot) > 0
            ), "The Bfr1 and Bfr2 masks should be different."
            if scores == "sum_by_volume":
                bfr1_size = np.count_nonzero(bfr1_spot)
                bfr2_size = np.count_nonzero(bfr2_spot)
                output = (
                    (map_.sum(where=bfr1_spot) / bfr1_size if bfr1_size else 0)
                    - (map_.sum(where=bfr2_spot) / bfr2_size if bfr2_size else 0)
                )
            else:
                output = map_.sum(where=bfr1_spot) - map_.sum(where=bfr2_spot)
            logging.info(
                f"hotspot {i + 1} {'dimer ' if by_dimers else 'chain '}"
                f"{(dimer_names if by_dimers else string.ascii_uppercase)[j]} "
                f"value {output:.4} ({output.dtype})"
            )
            result[i, j] = output
    # transpose the output for compatibility with analysis methods
    return result.transpose()


# Main script: parse and handle the optional arguments
if __name__ == "__main__":
    # Parse optional arguments.
    parser = argparse.ArgumentParser(description="Process and analyze an MRC map.")
    parser.add_argument(
        "map_file",
        nargs="?",  # zero or one
        default="284postprocess.mrc",
        help="MRC file of the experimental cryo-EM map",
    )
    parser.add_argument(
        "hotspot_file",
        nargs="?",
        default="bbRefinedHotSpotsListDaniel.csv",
        help="CSV file containing hotspot information",
    )
    parser.add_argument(
        "--divide_by_volume"
        # Todo
    )
    parser.add_argument(
        "--by_chain",
        dest="by_dimers",
        action="store_false",
        help="analyze the map chain by chain (default: by dimer)",
    )
    parser.add_argument(
        "--scores",
        choices=("sum", "sum_by_volume", "threshold"),
        default="sum",
        help="use threshold scoring or sum of densities scaled by volume (default: sum of densities unscaled)",
    )
    # threshold scoring does not give plausible results for homooligomeric structures
    parser.add_argument(
        "--homogeneous",
        action="store_true",
        help="add data points for homogeneous structures (default: no)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    out = process(
        args.hotspot_file, args.map_file, by_dimers=args.by_dimers, scores=args.scores
    )
    kargs = {"axis": 0, "keepdims": True}  # keepdims to keep a two-dimensional array
    if args.homogeneous:
        all_bfr1 = process(
            args.hotspot_file,
            "molmap_Bfr1_284_raz-rerefined-correction-res4.mrc",
            by_dimers=args.by_dimers,
            scores=args.scores,
        ).mean(**kargs)
        all_bfr2 = process(
            args.hotspot_file,
            "molmap_Bfr2_284_raz-rerefined-correction-res4.mrc",
            by_dimers=args.by_dimers,
            scores=args.scores,
        ).mean(**kargs)
        analysis.analyze(out, all_bfr1=all_bfr1, all_bfr2=all_bfr2)
    else:
        analysis.analyze(out)
