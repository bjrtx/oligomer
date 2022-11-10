import skimage
import mrcfile
import numpy as np
import hotspots
import shutil
import logging

"""
This script is not intended to be run frequently. It contains a method for taking one MRC file containing
12 heme positions and splitting it into 12 maps, one per location.
Each map is then associated with its corresponding dimers.
Finally one can prepare maps of the following form:
- a chain plus the location of its heme
- a dimer plus the location of its heme

The script needs to be run in the folder containing
- the chain maps, named as in the variable chain_filename below
- the heme map, named as in the variable heme_filename.

The outputs will be named "chain_plus_heme_A.mrc" and so on, "dimer_plus_heme_AQ.mrc" and so on.
"""

# small threshold to erase background noise when reading hemes
threshold = 0.0001

# in  the next variable, '?' will range from A to X
chain_file_name = "Bfr_molmap_chain?_res5.mrc"
heme_filename = "Heme_rerefined_corrected.mrc"
heme = hotspots.read_mrc(heme_filename)
if heme.dtype != np.float32:
    logging.warning("Read a file with unexpected numerical data type: {heme_filename}, {heme.dtype}.")

logging.info(
"size of all hemes after threshold", np.count_nonzero(heme > threshold)
)

heme_components = skimage.measure.label(heme > threshold)
assert set(heme_components.flat) == set(range(13))
# there are 12 connected components (1-12) plus 0 for empty regions

dimers = ["aq", "bo", "cv", "du", "ep", "fr", "gk", "hn", "is", "jt", "lw", "mx"]

for ab in dimers:
    chains = [
        hotspots.read_mrc(chain_file_name.replace('?', char.upper()))
        for char in ab
    ]

    intersection = heme_components[chains[0].astype(bool)]
    assert len(set(intersection.flat)) == 2  # one connected component label and 0
    label = intersection.max()

    # Part of the heme where heme_components == label
    right_component = np.where(heme_components == label, heme, np.float32(0))
    logging.info("Processing a heme of size {np.count_nonzero(right_component)}.")
    # Add the heme component to the chain by taking maxima element-wise.
    aug_chains = [np.maximum(chain, right_component) for chain in chains]
    dimer = np.maximum(*aug_chains)

    for char, chain in zip(ab, aug_chains):
        # preserve the headers
        shutil.copyfile(chain_file_name.replace('?', char.upper()), f"chain_plus_heme_{char.upper()}.mrc")
        with mrcfile.open(f"chain_plus_heme_{char.upper()}.mrc", 'r+') as file:
            file.set_volume() # chimera is doing something wrong here?
            file.set_data(chain)

    shutil.copyfile(chain_file_name.replace('?', char.upper()), f"dimer_plus_heme_{ab.upper()}.mrc")
    with mrcfile.open(f"dimer_plus_heme_{ab.upper()}.mrc", 'r+') as file:
        file.set_volume()
        file.set_data(dimer)
