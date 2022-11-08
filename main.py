import learning
import simulation
import hotspots

if __name__ == "__main__":
    # There must be a better way.
    map_filename = "284postprocess.mrc"
    map_threshold = 0.04
    hotspot_filename = "bbRefinedHotSpotsListDaniel.csv"
    out = hotspots.process(hotspot_filename, map_filename, map_threshold)
    learning.analyze(out.transpose())
