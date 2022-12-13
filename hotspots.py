from __future__ import annotations  # should avoid some typing problems in Python 3.8

import csv
import operator
import string
import logging
import heapq
import functools
import argparse
from collections.abc import Collection

import numpy as np
import skimage
import mrcfile

import analysis


def read_mrc(filename: str, dtype: type | None = None) -> np.ndarray:
    """
    Read an MRC map from a file and convert it into a Numpy multidimensional array.
    The values are cast to the specified data type if one is given.
    """
    with mrcfile.open(filename) as mrc:
        return mrc.data if dtype is None else mrc.data.astype(dtype)


def gloves(hotspot: dict[str]) -> list[np.ndarray, np.ndarray]:
    """
    Compute the two gloves associated with a given hotspot.

    hotspot: dictionary with the same fields as those expected by the method process
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
            logging.warning(
                f"One glove was empty: {hotspot}. This will create more warnings."
            )
    return gloves


def biggest_blob(logical: "np.ndarray[bool]", n: int = 1) -> "np.ndarray[bool]":
    """
    Given a logical multidimensional array, find the n largest connected regions
    and return them as a logical array. By default n is 1.
    If n is higher the elementwise maximum of the arrays is returned.
    If the input array is all-zero then so is the output.
    """
    labeled = skimage.measure.label(logical)
    regions = skimage.measure.regionprops(labeled)
    # regions is a list of connected components.
    # We select its n largest components.
    biggest = heapq.nlargest(n, regions, key=operator.attrgetter("area"))
    if len(biggest) < n:
        logging.warning(
            f"Fewer connected components than expected ({len(biggest)}/{n}), "
            f"perhaps due to an empty glove."
        )
    # return the union (logical sum) of all n largest components
    return functools.reduce(np.logical_or, (labeled == r.label for r in biggest), 0)


dimer_names = ["aq", "bo", "cv", "du", "ep", "fr", "gk", "hn", "is", "jt", "lw", "mx"]


def process(
    hotspot_data: str | Collection[dict[str]],
    map_: str | np.ndarray,
    by_dimers: bool = False,
    truncate: bool = True,
    gloves_data: Collection[tuple[np.ndarray, np.ndarray]] | None = None,
    scores: string = "sum",
):
    """
    Process a density map and return a 2-dimensional array (rows are hotspots,
    columns are chains, entries are scores). Alternatively, if by_dimers is True
    then the columns are dimers.

    hotspot_filename: either a CSV file or a list of rows from a previously read file.
        The CSV file contains hotspot information. It _must_ be saved
        in the UTF8 encoding. It has at least the following named columns:
        - "Bfr1_file_name" and "Bfr2_file_name" contain the paths of MRC files,
        - "Bfr1_Molmap_TH" and "Bfr2_Molmap_TH" contain floating point numbers
          between 0 and 1 intended as thresholds.

    map_filename: either the patn of an MRC file containing an electronic density map.

    truncate: when this is set to truethe maps are cast to 16-bit floats, shortening
    computation time.

    gloves_data: either a collection of already computed hotspot gloves (3-dimensional
    arrays of Boolean values) or None.

    scores: whether to score by sum of electronic density ("sum", the default behaviour)
    or by number of above-threshold values ("threshold")
    """
    assert scores in ("sum", "threshold")
    
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
    filenames = f"{'dimer' if by_dimers else 'chain'}_plus_heme_?.mrc"
    indices = dimer_names if by_dimers else string.ascii_uppercase[:24]
    columns = [
        read_mrc(filenames.replace("?", idx)) > chain_threshold for idx in indices
    ]
    out = np.empty([len(hotspot_data), len(columns)], dtype=np.float16)

    if scores == "threshold":
        map_ = (map_ > 0.025).astype(np.float16)  # for future precision printing
        logging.warning("Using threshold scoring.")

    for i, hotspot in enumerate(hotspot_data):
        bfr1, bfr2 = gloves(hotspot) if gloves_data is None else gloves_data[i]
        for j, chain_or_dimer in enumerate(columns):
            # To avoid parts of hotspots coming from other chains, only the
            # largest connected component of the hotspot-chain intersection
            # is kept.
            n_blobs = 2 if by_dimers else 1
            bfr1_spot = biggest_blob(bfr1 & chain_or_dimer, n_blobs)
            bfr2_spot = biggest_blob(bfr2 & chain_or_dimer, n_blobs)
            assert np.count_nonzero(bfr1_spot ^ bfr2_spot) > 0
            output = map_.sum(where=bfr1_spot) - map_.sum(
                where=bfr2_spot
            )
            logging.info(
                f"hotspot {i + 1} {'dimer ' if by_dimers else 'chain '}"
                f"{(dimer_names if by_dimers else string.ascii_uppercase)[j]} "
                f"value {output:.4} ({output.dtype})"
            )
            out[i, j] = output
    # transpose the output for compatibility with analysis methods
    return out.transpose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and analyze an MRC map.")
    parser.add_argument("map_file", nargs="?", default="284postprocess.mrc",
                        help="MRC file containing the map")
    parser.add_argument("hotspot_file", nargs="?",
                        default="bbRefinedHotSpotsListDaniel.csv",
                        help="CSV file containing hotspot information")
    parser.add_argument('--by_chain', dest='by_dimers', action='store_false',
                        help='analyze the map chain by chain (default: by dimer)')
    parser.add_argument('--scores', choices=("sum", "threshold"), default="sum",
                        help='use threshold scoring (default: sum of densities)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    out = process(args.hotspot_file, args.map_file, by_dimers=args.by_dimers,
                  scores=args.scores)
    kargs = {'axis': 0, 'keepdims': True}  # keepdims to keep a two-dimensional array
    all_bfr1 = process(args.hotspot_file, "molmap_Bfr1_284_raz-rerefined-correction-res4.mrc",
                       by_dimers=args.by_dimers, scores=args.scores).mean(**kargs)
    all_bfr2 = process(args.hotspot_file, "molmap_Bfr2_284_raz-rerefined-correction-res4.mrc",
                       by_dimers=args.by_dimers, scores=args.scores).mean(**kargs)
    print("out", out, "all_bfr1", all_bfr1, "all_bfr2", all_bfr2)
    analysis.analyze(out, all_bfr1=all_bfr1, all_bfr2=all_bfr2)
