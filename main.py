import sys

import learning
import simulation
import hotspots

if __name__ == "__main__":
    if len(sys.argv) != 3 or not sys.argv[1].endswith("mrc") or not sys.argv[2].endswith("csv"):
        print("Expected arguments: main.py map_filename hotspot_filename.")
    _, map_filename, hotspot_filename = sys.argv
    map_threshold = 0.04  #This does not seem important
    out = hotspots.process(hotspot_filename, map_filename, map_threshold)
    learning.analyze(out.transpose())
