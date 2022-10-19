import numpy
from sklearn.cluster import KMeans, SpectralClustering

# You need to be in the directory containing those files.
for fname in (
    "OUT_above_TH_by_chain_dom1.csv",
    "OUT_above_TH_by_chain_comb.csv",
    "OUT_sum_by_chain_dom1.csv",
    "OUT_sum_by_chain_comb.csv",
):
    data = numpy.loadtxt(fname, delimiter=",").transpose()
    for method in (KMeans, SpectralClustering):
        labels = method(n_clusters=2).fit(data).labels_
        #by convention, the first data point will always be in cluster 0
        if labels[0]:
            labels = 1 - labels
        print(labels)