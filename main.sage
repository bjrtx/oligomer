#!/usr/bin/env sage

# This script must be run with SageMath
# For example, using https://sagecell.sagemath.org/

from itertools import combinations
from collections import Counter
from sage.misc.banner import require_version
require_version(9, 6) # because of a bug in SageMath v9.5

def cuboctahedralGraph():
    """Returns an immutable copy of the cuboctahedral graph."""
    return polytopes.cuboctahedron().graph().copy(immutable=True)

g = cuboctahedralGraph()
assert(g.order() == 12 and g.num_edges() == 24
       and g.is_polyhedral() and g.is_regular(k=4))
assert(g.is_immutable())

G = g.automorphism_group()
assert(G.order() == 48)
# the automorphism group of g is (isomorphic to) S_4 * S_2 and can be interpreted as
# the full octahedral symmetry group (S_4 = O for rotations and S_2 for reflections)
H, *_ = SymmetricGroup(4).direct_product(SymmetricGroup(2))
assert(H.is_isomorphic(G))


#directed version, with arcs oriented =||
def directedCuboctahedralGraph():
    """Returns an immutable copy of the cuboctahedral graph
    with a specific edge orientation.
    """
    out_neighbours = {
        0: [1, 11],
        1: [2, 8],
        2: [3, 9],
        3: [0, 10],
        4: [7, 8],
        5: [4, 9],
        6: [10, 5],
        7: [6, 11],
        8: [5, 0],
        9: [1, 6],
        10: [2, 7],
        11: [3, 4]
    }
    return DiGraph(out_neighbours, immutable=True)

dg = directedCuboctahedralGraph()
assert(dg.to_undirected().is_isomorphic(g))
dG = dg.automorphism_group()
assert(dG.order() == 24)
# the automorphism group of dg is (isomorphic to) S_4 and can be interpreted as
# the rotational octahedral symmetry group (O, 432, etc.)
assert(SymmetricGroup(4).is_isomorphic(dG))

def nb_adjacencies(g, subset):
    """Number of pairs of vertices in subset which are adjacent in g."""
    return sum(g.has_edge(x, y) or g.has_edge(y, x) for x, y in combinations(subset, 2))

def plot(blue_set):
    def diamond(x, y, idx):
        p = polygon([(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)], color= (0, 0, 1) if idx in blue_set else (1, 0, 0), edgecolor= (0, 0, 0))
        if y == 1:
            p += line([(x - 0.5, y - 0.5),(x + 0.5, y + 0.5)], rgbcolor = (0, 0, 0))
        else:
            p += line([(x - 0.5, y + 0.5),(x + 0.5, y - 0.5)], rgbcolor = (0, 0, 0))
        return p
        
    centers = {9: (0, 0), 10: (0, 2), 2: (1,1), 1: (2, 0), 3: (2, 2), 0: (3, 1), 8: (4, 0), 11:(4,2), 4: (5, 1), 5: (6, 0), 7: (6, 2), 6: (7, 1)}
    sum(diamond(x,y, idx) for idx, (x, y) in centers.items()).show(axes=False)
    

def unique_colourings(nb_blue_vertices=6, directed=True):
    """List the colourings with a given number of blue vertices,
    either in the directed graph, up to rotations (if directed is True),
    or in the undirected graph, up to rotations and reflexions.
    """
    g = directedCuboctahedralGraph() if directed else cuboctahedralGraph()
    assert(0 <= nb_blue_vertices <= 12)
    unique_colourings = dict()
    for blue_set in combinations(g.vertices(), nb_blue_vertices):
        red_set = [v for v in g.vertices() if v not in blue_set]
        partition = [blue_set, red_set]
        # select a canonical representative of the (di)graph
        # up to automorphisms that preserve the red and blue colour classes
        # and either the edges or the directed edges
        canon = g.canonical_label(partition)
        if canon not in unique_colourings:
            auto = g.automorphism_group(partition=partition)
            adj = nb_adjacencies(g, red_set)    
            unique_colourings[canon] = (auto, adj)
            plot(blue_set)
    t = Counter((auto.order(), adj) for G, (auto, adj) in unique_colourings.items())
    print('{} case. With {} blue tiles there are {} unique colourings up to rotations.'.format(
          'Directed' if directed else 'Undirected', nb_blue_vertices, len(unique_colourings)), 
          'Order of the automorphism group / red-red adjacencies, number of such colourings',
          [k + (v,) for k, v in t.items()],
          """The smallest possible number of red-red adjacencies is {}.""".format(
          min(adj for _, adj  in unique_colourings.values())),
          sep='\n')
          
if __name__ == '__main__':
    unique_colourings()
