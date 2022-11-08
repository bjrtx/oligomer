import sys

import learning
import hotspots
import string
import numpy
import sklearn

def blue_dimers(map_filename, hotspot_filename):

    data = hotspots.process(hotspot_filename, map_filename, 0.04).transpose()
    chain_names = string.ascii_uppercase[:24]  # list of letters A to X inclusive
    shuffler = [
    (0, 16),
    (1, 14),
    (2, 21),
    (3, 20),
    (4, 15),
    (5, 17),
    (6, 10),
    (7, 13),
    (8, 18),
    (9, 19),
    (11, 22),
    (12, 23),
]
    first, second = zip(*shuffler)
    data = numpy.hstack((data[first, :], data[second, :]))
    chain_names = [
        chain_names[a] + chain_names[b] for a, b in shuffler
    ]
    labels = sklearn.cluster.KMeans(n_clusters=2).fit(data).labels_
    # by convention, the first data point will always be in cluster 0
    labels = labels != labels[0]
    return [name for name, label in zip(chain_names, labels) if label]



if __name__ == "__main__":
    if (
        len(sys.argv) != 3
        or not sys.argv[1].endswith("mrc")
        or not sys.argv[2].endswith("csv")
    ):
        print(
            """
Expected arguments: main.py map_filename hotspot_filename.

map_filename is an MRC file.
hotspot_filename is a CSV file in the UTF8 encoding.
"""
        )
        exit()
    _, map_filename, hotspot_filename = sys.argv
    map_threshold = 0.04  # This does not seem important
    out = hotspots.process(hotspot_filename, map_filename, map_threshold)
    learning.analyze(out.transpose())
