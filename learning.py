import numpy
from sklearn.cluster import KMeans

# You need to be in the directory containing those files.
for fname in (
    "OUT_above_TH_by_chain_dom1.csv", 
    "OUT_above_TH_by_chain_comb.csv",
    "OUT_sum_by_chain_dom1.csv",
    "OUT_sum_by_chain_comb.csv"
):
    data = numpy.loadtxt(fname, delimiter=",").transpose()
    print(KMeans(n_clusters=2).fit(data).labels_)
