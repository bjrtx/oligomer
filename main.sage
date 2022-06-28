#!/usr/bin/env sage

import sys 
from sage.all import *
from itertools import combinations

g = polytopes.cuboctahedron().graph()
assert(g.order() == 24 and g.num_edges() == 12 and g.is_polyhedral() and g.is_regular(k=4))
assert(g.automorphism_group().order() == 48)

nb_blue = 6 # between 0 and 12

def red_set(blue_set):
    return [v for v in g.vertices if v not in blue_set]

def nb_red_adjacencies(g, red_set):
    return sum(g.has_edge(x, y) for x, y in combinations(red_set, 2))

unique_colourings = {g.canonical_label(partition=[blue_set, red_set(blue_set)]) 
                     for blue_set in combinations(g.vertices(), nb_blue)}
print('There are', len(unique_colourings), 'unique colourings up to rotations and reflexions.')



information = [(G, G.automorphism_group().order()) for G in unique_colourings]
