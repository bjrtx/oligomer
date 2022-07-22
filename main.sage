#!/usr/bin/env sage

# This script must be run with SageMath
# For example, using https://sagecell.sagemath.org/

from itertools import combinations
from functools import cached_property
from collections import Counter, namedtuple
from sage.misc.banner import require_version
require_version(9, 6) # because of an apparent bug in SageMath v9.5

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

# Several assertions concerning this directed graph
if __name__ == '__main__':
    dg = directedCuboctahedralGraph()
    # The undirected graph underlying dg is the cuboctahedral graph
    assert(dg.to_undirected().is_isomorphic(polytopes.cuboctahedron().graph()))
    dG = dg.automorphism_group()
    assert(dG.order() == 24)
    # The automorphism group of dg is (isomorphic to) S_4 and can be interpreted as
    # the rotational octahedral symmetry group (O, 432, etc.)
    assert(SymmetricGroup(4).is_isomorphic(dG))

def nb_adjacencies(g, left, right):
    """Number of edges from left to right in the directed graph g."""
    return sum(g.has_edge(x, y) for x in left for y in right)

def plot(blue_set, blue=(0, 0, 1), red=(1, 0, 0)):
    """A Sage Graphics object representing an oligomer with coloured dimers."""
    blue_set = frozenset(blue_set)
    def diamond(x, y, idx):
        return polygon([(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)], color=(blue if idx in blue_set else red), edgecolor= (0, 0, 0)) \
               + line([(x - 0.5, y - 0.5),(x + 0.5, y + 0.5)] if y == 1 else [(x - 0.5, y + 0.5),(x + 0.5, y - 0.5)], rgbcolor = (0, 0, 0))
        
    centers = {9: (0, 0), 10: (0, 2), 2: (1,1), 1: (2, 0), 3: (2, 2), 0: (3, 1), 8: (4, 0), 11:(4,2), 4: (5, 1), 5: (6, 0), 7: (6, 2), 6: (7, 1)}
    return sum(diamond(x,y, idx) for idx, (x, y) in centers.items())

Adjacencies = namedtuple('Adjacencies', ['BB', 'RR', 'BR', 'RB'])
Adjacencies.__repr__ = lambda self: f'junctions: blue->orange {self.BR}, blue->blue {self.BB}, orange->orange {self.RR}, orange->blue {self.RB}'
    

class Bicolouring:
    """A class that stores information about a blue-red colouring of a (directed) graph."""
    
    def __init__(self, graph, blue_set):
        self.graph = graph
        self.blue_set = frozenset(blue_set)
        self.red_set = frozenset(v for v in graph.vertices() if v not in blue_set)
    
    @cached_property
    def canon(self):
        """A label that identifies the colouring up to automorphisms."""
        return self.graph.canonical_label([self.blue_set, self.red_set])
    
    @cached_property
    def adjacencies(self):
        return Adjacencies(nb_adjacencies(self.graph, self.blue_set, self.blue_set),
                           nb_adjacencies(self.graph, self.red_set, self.red_set),
                           nb_adjacencies(self.graph, self.blue_set, self.red_set),
                           nb_adjacencies(self.graph, self.red_set, self.blue_set))
    
    @cached_property
    def automorphism_group(self):
        return self.graph.automorphism_group(partition=[self.blue_set, self.red_set])
    
    def __repr__(self):
        return f'{len(self.blue_set)}:{len(self.red_set)}, {self.adjacencies}, automorphism group of order {self.automorphism_group.order()}'
    
    @cached_property
    def picture(self):
        """Only works properly if self.graph is the directed cuboctahedral graph."""
        return plot(self.blue_set)
    
    def show(self):
        """Displays a picture of the colouring."""
        self.picture.show(axes=False)
        

def unique_colourings(nb_blue_vertices=6, graph=None, show=False):
    """List the colourings with a given number of blue vertices,
    in the directed graph, up to rotations.
    """
    graph = directedCuboctahedralGraph() if graph is None else graph
    assert(0 <= nb_blue_vertices <= graph.order())
    unique_colourings = {}
    for blue_set in combinations(graph.vertices(), nb_blue_vertices):
        c = Bicolouring(graph, blue_set)
        # select a canonical representative of the digraph
        # up to automorphisms that preserve the red and blue colour classes
        # and the directed edges
        if c.canon not in unique_colourings:
            unique_colourings[c.canon] = c
            if show: c.show()
    return unique_colourings

def short_display(nb_blue_vertices, **options):
    a = unique_colourings(nb_blue_vertices=nb_blue_vertices, **options)
    print(f'{nb_blue_vertices}:{12 - nb_blue_vertices}, the number of distinct arrangements is {len(a)}')
    C = Counter(repr(c) for c in sorted(a.values(), key = lambda c:c.adjacencies.BR, reverse=True))
    for v, k in C.items():
        print(v + f'\t({k} such arrangements)' if k > 1 else v)
    
          
if __name__ == '__main__':
    short_display(6)
    print('-' * 100)
    short_display(7)
