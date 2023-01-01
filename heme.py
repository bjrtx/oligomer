from __future__ import annotations

import shutil
import logging

import skimage
import mrcfile
import numpy as np
import hotspots

"""
This script is not intended to be run frequently. It contains a method for taking one
MRC file containing 12 heme positions and splitting it into 12 maps, one per location.
Each map is then associated with its corresponding dimers. Finally one can prepare maps
of the following form:
- a chain plus the location of its heme,
- a dimer plus the location of its heme.

The script needs to be run in the folder containing
- the chain maps, named as in the variable chain_filename below,
- the heme map, named as in the variable heme_filename.

The outputs will be named "chain_plus_heme_A.mrc" and so on, "dimer_plus_heme_AQ.mrc"
and so on.
"""

logging.basicConfig(level=logging.INFO)

# small threshold to erase background noise when reading hemes
threshold = 0.00001

# in  the next variable, '?' will range from A to X
chain_file_name = "Bfr_molmap_chain?_res5.mrc"
heme_filename = "Heme_rerefined_corrected.mrc"

heme = hotspots.read_mrc(heme_filename)
heme = np.where(heme > threshold, heme, 0)  # thresholding the heme map

if heme.dtype != np.float32:
    logging.warning(
        "Read a file with unexpected data type: {heme_filename}, {heme.dtype}. "
        "Expected numpy.float32."
    )

logging.info(f"Size of the thresholded heme map: {np.count_nonzero(heme)}")

heme_components = skimage.measure.label(heme.astype(bool))
# There are 12 connected components (1-12) plus 0 for empty regions
assert set(heme_components.flat) == set(range(13))

example_file = chain_file_name.replace("?", "A")

# dimers = ["aq", "bo", "cv", "du", "ep", "fr", "gk", "hn", "is", "jt", "lw", "mx"]
for ab in hotspots.dimer_names:
    logging.info(f"Writing maps for the chains in {ab}.")
    chains = [
        hotspots.read_mrc(chain_file_name.replace("?", char.upper())) for char in ab
    ]
    # There is exactly one nonzero label in the intersection of chains[0] and the heme.
    label = max(labels := set(heme_components[chains[0].astype(bool)].flat))
    assert (
        labels == {0, label} and 1 <= label <= 12
    )  # one connected component label and 0

    right_component = np.where(heme_components == label, heme, 0)
    logging.info(f"Processing a heme of size {np.count_nonzero(right_component)}.")
    # Add the heme component to the chain by taking maxima element-wise.
    aug_chains = [np.maximum(chain, right_component) for chain in chains]
    dimer = np.maximum(*aug_chains)

    for char, chain in zip(ab, aug_chains):
        # Copy an existing file to preserve the header. MRC headers have many parameters
        # to set. (In fact, Chimera seems to omit some of them.)
        new_file = f"chain_plus_heme_{char.upper()}.mrc"
        shutil.copyfile(example_file, new_file)
        with mrcfile.open(new_file, "r+") as file:
            file.set_volume()  # Chimera seems not to set its headers properly
            file.set_data(chain)

    new_file = f"dimer_plus_heme_{ab.upper()}.mrc"
    shutil.copyfile(example_file, new_file)
    with mrcfile.open(new_file, "r+") as file:
        file.set_volume()
        file.set_data(dimer)
