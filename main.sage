#!/usr/bin/env sage

# This script must be run with SageMath
# See installation instructions at https://www.sagemath.org/
# or use an online interpreter such as https://sagecell.sagemath.org/

from __future__ import annotations
from itertools import chain, combinations
from functools import cached_property
from collections import Counter, namedtuple
from collections.abc import Iterable
from dataclasses import dataclass

import csv
import sys

# colours, including those matching UCSF Chimera's
BLACK = (0, 0, 0)
RED = (1, 0, 0)
CHIMERA_RED = (0.776, 0.208, 0.031)
BLUE = (0, 0, 1)
CHIMERA_BLUE = (0.051, 0.02, 0.933)

# type aliases
Graph = sage.graphs.generic_graph.GenericGraph
DiGraph = sage.graphs.digraph.DiGraph

# directed version, with arcs oriented =||
def directed_cuboctahedral_graph() -> DiGraph:
    """Return an immutable copy of the cuboctahedral graph with a specific edge orientation.
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
        11: [3, 4],
    }
    # fixing vertex positions for drawing the graph
    pos = [
        [-1, -1],
        [-1, 1],
        [1, 1],
        [1, -1],
        [-5, -5],
        [-5, 5],
        [5, 5],
        [5, -5],
        [-3, 0],
        [0, 3],
        [3, 0],
        [0, -3],
    ]
    return DiGraph(out_neighbours, pos=dict(enumerate(pos)), immutable=True)

def vertices_to_facets():
    """Return a dict mapping each vertex of the directed octahedral graph to a facet of
    the rhombic dodecahedron."""
    facet_dict = {i: f for i, f in enumerate(polytopes.rhombic_dodecahedron().facets())}
    edges = [
        (i, j) for (i, f), (j, g) in combinations(face_dict.items(), 2) 
        if f.as_polyhedron().intersection(g.as_polyhedron()).dimension() == 1
        ]
    assert len(edges) == 24 and len(facets) == 12
    g = sage.graphs.graph.Graph(data=edges)
    h = directed_cuboctahedral_graph().to_undirected()
    _, mapping = g.is_isomorphic(h, certificate=True)
    return {mapping[i]: f for i, f in face_dict.items()}


def more_complicated_graph() -> DiGraph:
    out_neighbours = {
        0: [(1, 1), (11, 0)],
        1: [(2, 0), (8, 0)],
        2: [(3, 0), (9, 1)],
        3: [(0, 1), (10, 1)],
        4: [(7, 0), (8, 1)],
        5: [(4, 0), (9, 0)],
        6: [(10, 0), (5, 1)],
        7: [(6, 1), (11, 1)],
        8: [(5, 0), (0, 0)],
        9: [(1, 0), (6, 0)],
        10: [(2, 1), (7, 1)],
        11: [(3, 1), (4, 1)],
    }
    out_neighbours = ([((k, 0), v), ((k, 1), v)] for k, v in out_neighbours.items())
    out_neighbours = dict(chain.from_iterable(out_neighbours))
    return DiGraph(out_neighbours, immutable=True)


# Several assertions concerning this directed graph
if __name__ == "__main__":
    dg = directed_cuboctahedral_graph()
    assert dg.is_isomorphic(
        sage.all.AlternatingGroup(4).cayley_graph(), edge_labels=False
    )
    assert dg.to_undirected().is_isomorphic(sage.all.polytopes.cuboctahedron().graph())
    # The automorphism group of dg is (isomorphic to) S_4 and can be interpreted as the
    # rotational octahedral symmetry group (O, 432, etc.).
    assert sage.all.SymmetricGroup(4).is_isomorphic(dg.automorphism_group())


def nb_adjacencies(graph: Graph, left: Iterable, right: Iterable) -> int:
    """Count the edges from left to right in the directed graph."""
    return sum(graph.has_edge(x, y) for x in set(left) for y in set(right))


def _plot(blue_set: Iterable, blue=CHIMERA_BLUE, red=CHIMERA_RED):
    """Return a Sage Graphics object representing an oligomer with coloured
    dimers.
    """
    blue_set = frozenset(blue_set)

    def diamond(center, idx):
        x, y = center
        return sage.all.polygon(
            [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)],
            color=(blue if idx in blue_set else red),
            edgecolor=BLACK,
        ) + sage.all.line(
            [(x - 0.5, y - 0.5), (x + 0.5, y + 0.5)]
            if y == 1
            else [(x - 0.5, y + 0.5), (x + 0.5, y - 0.5)],
            rgbcolor=BLACK,
        )

    centers = {
        9: (0, 0),
        10: (0, 2),
        2: (1, 1),
        1: (2, 0),
        3: (2, 2),
        0: (3, 1),
        8: (4, 0),
        11: (4, 2),
        4: (5, 1),
        5: (6, 0),
        7: (6, 2),
        6: (7, 1),
    }

    return sum(diamond(center, idx) for idx, center in centers.items())


Adjacencies = namedtuple("Adjacencies", ["BB", "RR", "BR", "RB"])
Adjacencies.__str__ = lambda self: (
    f"junctions: blue->orange {self.BR}, blue->blue {self.BB}, "
    f"orange->orange {self.RR}, orange->blue {self.RB}"
)


@dataclass(frozen=True)
class Bicolouring:
    """Store information about a blue-red colouring of a (directed) graph.
    Two such colourings that are equivalent up to automorphism will test equal
    and will have the same hash.
    """

    graph: Graph = directed_cuboctahedral_graph()
    blue_set: frozenset[int] = frozenset()

    def __post_init__(self):
        object.__setattr__(
            self,
            "graph",
            self.graph
            if self.graph.is_immutable()
            else self.graph.copy(immutable=True),
        )
        object.__setattr__(self, "blue_set", frozenset(self.blue_set))
        vertices = frozenset(self.graph.vertices())
        object.__setattr__(self, "red_set", vertices - self.blue_set)

    @cached_property
    def _canon(self) -> Bicolouring:
        """Return an object that characterises the colouring up to automorphisms. In practice, this is a 
        relabelling of self.graph in which the vertices of self.blue_set come first. This relabelling is the
        same for two colourings of the same graph with a colour-preserving isomorphism.
        """
        canon, mapping = self.graph.canonical_label(partition=[self.blue_set, self.red_set], certificate=True)
        return Bicolouring(canon, blue_set={mapping[v] for v in self.blue_set})

    @cached_property
    def adjacencies(self):
        """Count the number of adjacencies, sorted by colours."""
        return Adjacencies(
            nb_adjacencies(self.graph, self.blue_set, self.blue_set),
            nb_adjacencies(self.graph, self.red_set, self.red_set),
            nb_adjacencies(self.graph, self.blue_set, self.red_set),
            nb_adjacencies(self.graph, self.red_set, self.blue_set),
        )

    @cached_property
    def automorphism_group(self):
        """Return the group of automorphisms that preserve vertex colours."""
        return self.graph.automorphism_group(partition=[self.blue_set, self.red_set])

    def __str__(self):
        return (
            f"{len(self.blue_set)}:{len(self.red_set)}, {self.adjacencies}, "
            f"symmetry number {self.automorphism_group.order()}"
        )

    def __eq__(self, other: Bicolouring):
        return self._canon.graph == other._canon.graph

    def __hash__(self):
        return hash(self._canon.graph)

    def distance(self, other: Bicolouring) -> int:
        """Count the vertices which have distinct colours in self and in other."""
        return len(self.blue_set.symmetric_difference(other.blue_set))

    def show(self, mode='net') -> None:
        """Displays a picture of the colouring."""
        if mode == 'net':
            _plot(self.blue_set).show(axes=False)
        elif mode == 'graph':
            self.graph.plot(
                vertex_labels = None,
                vertex_color = CHIMERA_RED,
                vertex_colors = {CHIMERA_BLUE: self.blue_set}
                ).show()
        elif mode == 'polyhedron':
            facets = colouring_to_facets()
            sum(facets[i].as_polyhedron().plot(polygon=CHIMERA_BLUE if i in self.blue_set else CHIMERA_RED) for i in range(12)).show(frame=False)



def unique_colourings(
    nb_blue_vertices=6,
    *,
    graph: Graph = directed_cuboctahedral_graph(),
    isomorphism=True,
) -> Iterable[Bicolouring]:
    """List the colourings with a given number of blue vertices in the directed graph,
    either up to rotations (if isomorphism is True) or not. Return either a list or a set.

    Keyword arguments:
    nb_blue_vertices -- number of vertices to be coloured in blue
    graph -- graph to be coloured
    isomorphism -- if True, the colourings are counted up to colour-preserving automorphisms
    of the graph. If False, all colourings are listed.
    """
    assert 0 <= nb_blue_vertices <= graph.order()
    return (set if isomorphism else list)(
        Bicolouring(graph, blue_set)
        for blue_set in combinations(graph.vertices(), nb_blue_vertices)
    )


def write_to_csv(
    colourings: Iterable[Bicolouring],
    csv_file=None,
    *,
    csv_header=True,
    dialect="excel",
):
    """Write the contents of colourings to csv_file.
    
    Keyword arguments:
    csv_file -- a path-like object (such as a string corresponding to a filename) or None.
    If None, the output will be written to the standard output.
    csv_header -- if True, add a header with column names as the first row
    dialect -- csv format, see Python csv documentation 
    """

    def write(file):
        writer = csv.writer(file, dialect=dialect)
        if csv_header:
            writer.writerow(
                [
                    "blue vertices",
                    "red vertices",
                    "blue->red",
                    "blue->blue",
                    "red->red",
                    "red->blue",
                    "symmetry number",
                ]
            )
        for colouring in colourings:
            writer.writerow(
                [
                    len(colouring.blue_set),
                    len(colouring.red_set),
                    colouring.adjacencies.BR,
                    colouring.adjacencies.BB,
                    colouring.adjacencies.RR,
                    colouring.adjacencies.RB,
                    colouring.automorphism_group.order(),
                ]
            )

    if csv_file:
        with open(csv_file, "a") as file:
            write(file)
    else:
        write(sys.stdout)


def short_display(
    nb_blue_vertices, csv_=False, graph: Graph = directed_cuboctahedral_graph(),
    **csv_options
):
    """Display the unique colourings for a given number of blue vertices.
    
    Keyword arguments:
    csv_ -- whether to output in the csv format
    csv_file, csv_header -- options passed to write_to_csv if csv_ is True
    """
    colourings = unique_colourings(nb_blue_vertices=nb_blue_vertices, graph=graph, **options)

    if csv_:
        write_to_csv(colourings, **csv_options)
    else:
        colourings = sorted(colourings, key=lambda c: c.adjacencies.BR, reverse=True)
        lines = Counter(str(c) for c in colourings)
        print(
            f"With {nb_blue_vertices} blue vertices and {graph.order() - nb_blue_vertices} orange vertices,"
            f" the number of distinct arrangements is {len(colourings)}."
        )
        for val, key in lines.items():
            print(val + f"\t({key} such arrangements)" if key > 1 else val)


def overlap() -> dict[Bicolouring]:
    """List the pairs (c1, c2) where c1 has 6 blue vertices, c2 has 7 blue
    vertices and c2 is obtained from c1 by changing a single vertex colour.
    """
    colourings1 = (
        c for c in unique_colourings(6, isomorphism=False) if c.adjacencies.BR == 8
    )
    colourings2 = [
        c for c in unique_colourings(7, isomorphism=False) if c.adjacencies.BR == 8
    ]
    return {
        c1: {c2 for c2 in colourings2 if c1.distance(c2) == 1} for c1 in colourings1
    }


def _experimental_colouring(switch=True) -> Bicolouring:
    """Return the colouring which the data seem to indicate, with the last vertex either
    red or blue as switch is True or False."""
    return Bicolouring(blue_set=[10, 2, 1, 11, 4, 5] + ([] if switch else [7]))


if __name__ == "__main__":
    for c in unique_colourings(2):
        c.show(mode='graph')
        c.show(mode='polyhedron')