#!/usr/bin/env sage


from itertools import combinations
from sage.misc.banner import require_version
require_version(9, 6)
# a bug in Sage 9.5 prevents its use

# the cuboctahedral graph
def cuboctahedralGraph():
    return polytopes.cuboctahedron().graph().copy(immutable=True)

g = cuboctahedralGraph()
assert(g.order() == 12 and g.num_edges() == 24
       and g.is_polyhedral() and g.is_regular(k=4))
assert(g.is_immutable())

G = g.automorphism_group()
assert(G.order() == 48)
H, *_ = SymmetricGroup(4).direct_product(SymmetricGroup(2))
assert(H.is_isomorphic(G))


#directed version, with arcs oriented =||
def directedCuboctahedralGraph():
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
assert(SymmetricGroup(4).is_isomorphic(dG))

def nb_adjacencies(g, subset):
    return sum(g.has_edge(x, y) or g.has_edge(y, x) for x, y in combinations(subset, 2))
    

def unique_colourings(nb_blue_vertices=6, directed=True):
    g = directedCuboctahedralGraph() if directed else cuboctahedralGraph()
    assert(0 <= nb_blue_vertices <= 12)
    unique_colourings = dict()
    for blue_set in combinations(g.vertices(), nb_blue_vertices):
        red_set = [v for v in g.vertices() if v not in blue_set]
        partition= [blue_set, red_set]
        # select a canonical representative of the (di)graph
        # up to automorphisms that preserve the red and blue colour classes
        canon = g.canonical_label(partition)
        if canon not in unique_colourings:
            auto = g.automorphism_group(partition=partition)
            adj = nb_adjacencies(g, red_set)    
            unique_colourings[canon] = (auto, adj)
    
    if directed: print('Directed case')
    print('With', nb_blue_vertices, 'blue tiles there are', 
          len(unique_colourings), 'unique colourings up to rotations and reflexions.')
    print('Symmetry / red adjacencies:')
    print([(auto.order(), adj) for G, (auto, adj) in unique_colourings.items()])
    print('The smallest possible number of red-red adjacencies is', 
          min(adj for _, adj  in unique_colourings.values()))
          
if __name__ == '__main__':
    unique_colourings()
