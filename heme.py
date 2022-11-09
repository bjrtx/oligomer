import skimage
import mrcfile
import numpy as np
import hotspots

"""
This script is not intended to be run frequently. It contains a method for taking one MRC file containing
12 heme positions and splitting it into 12 maps, one per location.
Each map is then associated with its corresponding dimers.
Finally one can prepare maps of the following form:
- a chain plus the location of its heme
- a dimer plus the location of its heme
"""

heme_filename = "Heme_rerefined_corrected.mrc"
heme = hotspots.read_mrc(
    heme_filename, dtype=np.float32
)  # no truncating the floats here

heme_components = skimage.measure.label(heme)
assert set(heme_components.flat) == set(range(13))
# there are 12 connected components (1-12) plus 0 for empty regions


dimers = ["aq", "bo", "cv", "du", "ep", "fr", "gk", "hn", "is", "jt", "lw", "mx"]

for ab in dimers:
    chains = [
        hotspots.read_mrc(f"Bfr_molmap_chain{char.upper()}_res5.mrc", dtype=np.float32)
        for char in ab
    ]

    intersection = heme_components[chains[0].astype(bool)]
    assert len(set(intersection.flat)) == 2  # one connected component label and 0
    label = intersection.max()

    # Part of the heme where heme_components == label
    right_component = np.where(heme, heme_components == label, 0)

    # Add the heme component to the chain by taking maxima element-wise.
    chains = [np.maximum(chain, right_component) for chain in chains]
    dimer = np.maximum(*chains)

    for char, chain in zip(ab, chains):
        mrcfile.write(
            f"chain_plus_heme_{char.upper()}.mrc",
            chain.astype(np.float32),
            overwrite=True,
        )

    mrcfile.write(
        f"dimer_plus_heme_{ab.upper()}.mrc", dimer.astype(np.float32), overwrite=True
    )
