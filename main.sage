#!/usr/bin/env sage


from itertools import combinations
from sage.misc.banner import require_version
require_version(9, 6)


# let g be the cuboctahedral graph
g = polytopes.cuboctahedron().graph().copy(immutable=True)
assert(g.order() == 12 and g.num_edges() == 24
       and g.is_polyhedral() and g.is_regular(k=4))
assert(g.is_immutable())
# the cuboctahedron has rotational symmetries (order 24) 
# and reflexive symmetry (order 2)
assert(g.automorphism_group().order() == 48)
print(g.automorphism_group().as_finitely_presented_group())



def nb_adjacencies(g, subset):
    return sum(g.has_edge(x, y) for x, y in combinations(subset, 2))
    
#def partition_to_polyhedron(blue, red):
    

def unique_colourings(nb_blue_vertices=6):
    assert(0 <= nb_blue_vertices <= 12)
    unique_colourings = dict()
    for blue_set in combinations(g.vertices(), nb_blue_vertices):
        red_set = [v for v in g.vertices() if v not in blue_set]
        partition= [blue_set, red_set]
        canon = g.canonical_label(partition)
        if canon not in unique_colourings:
            auto = g.automorphism_group(partition=partition)
            adj = nb_adjacencies(g, red_set)    
            unique_colourings[canon] = (auto, adj)
    
    print('With', nb_blue_vertices, 'blue tiles there are', 
          len(unique_colourings), 'unique colourings up to rotations and reflexions.')
    print('Symmetry / red adjacencies:')
    print([(auto.order(), adj) for G, (auto, adj) in unique_colourings.items()])
    print('The smallest possible number of red-red adjacencies is', 
          min(adj for _, adj  in unique_colourings.values()))
          
if __name__ == '__main__':
    unique_colourings()
