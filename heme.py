import skimage
import mrcfile
import numpy as np

"""
This script is not intended to be run frequently. It contains a method for taking one MRC file containing
12 heme positions and splitting it into 12 maps, one per location.
Each map is then associated with its corresponding dimers.
Finally one can prepare maps of the following form:
- a chain plus the location of its heme
- a dimer plus the location of its heme
"""
