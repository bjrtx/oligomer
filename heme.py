import skimage
import mrcfile
import numpy as np
import hotspots
import shutil

"""
This script is not intended to be run frequently. It contains a method for taking one MRC file containing
12 heme positions and splitting it into 12 maps, one per location.
Each map is then associated with its corresponding dimers.
Finally one can prepare maps of the following form:
- a chain plus the location of its heme
- a dimer plus the location of its heme
"""


threshold = 0.0001  # change threshold for smaller / larger hemes

heme_filename = "Heme_rerefined_corrected.mrc"
heme = hotspots.read_mrc(heme_filename)
print("heme", heme.dtype)

print("size of all hemes after threshold", np.count_nonzero(heme > threshold))

heme_components = skimage.measure.label(heme > threshold)
assert set(heme_components.flat) == set(range(13))
# there are 12 connected components (1-12) plus 0 for empty regions

dimers = ["aq", "bo", "cv", "du", "ep", "fr", "gk", "hn", "is", "jt", "lw", "mx"]

for ab in dimers:
    chains = [
        hotspots.read_mrc(f"Bfr_molmap_chain{char.upper()}_res5.mrc") for char in ab
    ]

    intersection = heme_components[chains[0].astype(bool)]
    assert len(set(intersection.flat)) == 2  # one connected component label and 0
    label = intersection.max()

    # Part of the heme where heme_components == label
    right_component = np.where(heme_components == label, heme, np.float32(0))
    print(
        "component dtype",
        right_component.dtype,
        right_component.shape,
        right_component.max(),
    )
    print("heme size", np.count_nonzero(right_component))
    # Add the heme component to the chain by taking maxima element-wise.
    aug_chains = [np.maximum(chain, right_component) for chain in chains]
    for c in aug_chains:
        print("chain", c.dtype, c.shape)
    dimer = np.maximum(*aug_chains)

    for char, chain in zip(ab, aug_chains):
        # preserve the headers
        shutil.copyfile(
            f"Bfr_molmap_chain{char.upper()}_res5.mrc",
            f"chain_plus_heme_{char.upper()}.mrc",
        )
        with mrcfile.open(f"chain_plus_heme_{char.upper()}.mrc", "r+") as file:
            file.set_volume()  # chimera is doing something wrong here?
            file.set_data(chain)

    shutil.copyfile(
        f"Bfr_molmap_chain{char.upper()}_res5.mrc", f"dimer_plus_heme_{ab.upper()}.mrc"
    )
    with mrcfile.open(f"dimer_plus_heme_{ab.upper()}.mrc", "r+") as file:
        file.set_volume()
        file.set_data(dimer)
