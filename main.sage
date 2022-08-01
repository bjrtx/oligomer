#!/usr/bin/env sage

# This script must be run with SageMath
# See installation instructions at https://www.sagemath.org/
# or use an online interpreter such as https://sagecell.sagemath.org/

from __future__ import annotations
from itertools import chain, combinations
from functools import cache, cached_property
from collections import Counter, namedtuple
from collections.abc import Iterable
from dataclasses import dataclass

import csv
import sys

# colors matching UCSF Chimera's
CHIMERA_RED = (0.776, 0.208, 0.031)
MEDIUM_BLUE = sage.plot.colors.Color("#3232cd").rgb()  # from Chimera

# type aliases
Graph = sage.graphs.generic_graph.GenericGraph
DiGraph = sage.graphs.digraph.DiGraph

@cache
def directed_cuboctahedral_graph() -> DiGraph:
    """Return an immutable copy of the cuboctahedral graph with a specific edge orientation,
    as described in the companion paper.
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
    return DiGraph(out_neighbours, pos=dict(enumerate(pos)), immutable=True, format='dict_of_lists')


@cache
def _vertices_to_dimers() -> dict[str]:
    """Return a dict mapping each vertex of the directed octahedral graph to the two letters
    designing its dimers (top then bottom) in the Chimera-generated net pictures.
    """
    return {
        10: "ob",
        9: "nh",
        2: "cv",
        3: "kg",
        1: "xm",
        0: "fr",
        11: "is",
        8: "ud",
        4: "tj",
        7: "wl",
        5: "pe",
        6: "qa",
    }


@cache
def _vertices_to_facets() -> dict:
    """Return a dict mapping each vertex of the directed octahedral graph to a facet of
    the rhombic dodecahedron."""
    facets = {i: f for i, f in enumerate(polytopes.rhombic_dodecahedron().facets())}
    edges = [
        (i, j)
        for (i, f), (j, g) in combinations(facets.items(), 2)
        if f.as_polyhedron().intersection(g.as_polyhedron()).dimension() == 1
    ]
    assert len(edges) == 24 and len(facets) == 12
    _, isomorphism = sage.graphs.graph.Graph(data=edges).is_isomorphic(
        directed_cuboctahedral_graph().to_undirected(), certificate=True
    )
    return {isomorphism[i]: f for i, f in facets.items()}


def oligomer_structure(blue_set: set = frozenset()):
    """Return a Sage Graphics object representing a 24-mer whose facets are colored according
    to blue_set.
    """
    facets = {i :f.as_polyhedron() for i, f in _vertices_to_facets().items()}
    g = directed_cuboctahedral_graph()
    struct = 0

    for i, f in facets.items():
        edges_out = [
            f.intersection(facets[j])
            for j in g.neighbors_out(i)
        ]
        midpoints = [e.center() for e in edges_out]
        edges_in = [
            f.intersection(facets[j])
            for j in g.neighbors_in(i)
        ]
        for e in edges_in:
            struct += Polyhedron(vertices=chain(e.vertices(), midpoints)).plot(
                line={"color": 'black', "thickness": 8},
                polygon=MEDIUM_BLUE if i in blue_set else CHIMERA_RED,
            )

    return struct


@cache
def more_complicated_graph() -> DiGraph:
    """Return a digraph that encodes dimer-dimer junctions (for the case of heterodimers)."""
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
    out_neighbours = (((k, i), v) for k, v in out_neighbours.items() for i in (0, 1))
    out_neighbours = dict(chain(out_neighbours))
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

Adjacencies = namedtuple("Adjacencies", ["BB", "RR", "BR", "RB"])
Adjacencies.__str__ = lambda self: (
    f"junctions: blue->orange {self.BR}, blue->blue {self.BB}, "
    f"orange->orange {self.RR}, orange->blue {self.RB}"
)


@dataclass(frozen=True)
class Bicoloring:
    """Store information about a blue-red coloring of a (directed) graph.
    Two such colorings that are equivalent up to automorphism will test equal
    and will have the same hash.
    """

    graph: Graph
    blue_set: frozenset = frozenset()

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
    def canonical_form(self) -> Bicoloring:
        """Return an object that characterises the coloring up to automorphisms. In practice, this is a
        relabelling of self.graph in which the vertices of self.blue_set come first. This relabelling is the
        same for two colorings of the same graph with a color-preserving isomorphism.
        """
        canon, mapping = self.graph.canonical_label(
            partition=[self.blue_set, self.red_set], certificate=True
        )
        return Bicoloring(canon, blue_set={mapping[v] for v in self.blue_set})

    def nb_adjacencies(self, left: Iterable, right: Iterable) -> int:
        """Count the edges from left to right."""
        return sum(self.graph.has_edge(x, y) for x in set(left) for y in set(right))

    @cached_property
    def adjacencies(self):
        """Count the number of adjacencies, sorted by colors."""
        return Adjacencies(
            self.nb_adjacencies(self.blue_set, self.blue_set),
            self.nb_adjacencies(self.red_set, self.red_set),
            self.nb_adjacencies(self.blue_set, self.red_set),
            self.nb_adjacencies(self.red_set, self.blue_set),
        )

    @cached_property
    def automorphism_group(self):
        """Return the group of automorphisms that preserve vertex colors."""
        return self.graph.automorphism_group(partition=[self.blue_set, self.red_set])

    def __str__(self):
        return (
            f"{len(self.blue_set)}:{len(self.red_set)}, {self.adjacencies}, "
            f"symmetry number {self.automorphism_group.order()}"
        )

    def __eq__(self, other: Bicoloring):
        return self.canonical_form.graph == other.canonical_form.graph

    def __hash__(self):
        return hash(self.canonical_form.graph)

    def distance(self, other: Bicoloring) -> int:
        """Count the vertices which have distinct colors in self and in other."""
        return len(self.blue_set.symmetric_difference(other.blue_set))

    def show(self) -> None:
        """Displays a picture of the coloring."""
        self.graph.plot(
            vertex_labels=None,
            vertex_color=CHIMERA_RED,
            vertex_colors={MEDIUM_BLUE: self.blue_set},
        ).show()


@dataclass(frozen=True, eq=False)  # inherit inequality from the parent class
class OctahedralBicoloring(Bicoloring):
    """Provide methods specific to bicolorings of the directed octahedral graph."""

    def __init__(self, blue_set: frozenset[int] = frozenset()):
        super().__init__(graph=directed_cuboctahedral_graph(), blue_set=blue_set)

    def show(self, mode="net") -> None:
        """Displays a picture of the coloring.

        Keyword arguments:
        mode -- one of "net", "graph", "polyhedron"
        """
        if mode == "net":

            def diamond(center, idx):
                x, y = center
                return sage.all.polygon(
                    [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)],
                    color=(MEDIUM_BLUE if idx in self.blue_set else CHIMERA_RED),
                    edgecolor='black',
                ) + sage.all.line(
                    [(x - 0.5, y - 0.5), (x + 0.5, y + 0.5)]
                    if y == 1
                    else [(x - 0.5, y + 0.5), (x + 0.5, y - 0.5)],
                    rgbcolor='black',
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

            sum(diamond(center, idx) for idx, center in centers.items()).show(
                axes=False
            )

        elif mode == "graph":
            super().show()
        elif mode == "polyhedron":
            facets = _vertices_to_facets()
            (
                # sum(
                #    facets[i]
                #    .as_polyhedron()
                #    .plot(polygon=MEDIUM_BLUE if i in self.blue_set else CHIMERA_RED)
                #    for i in range(12)
                # )
                # +
                oligomer_structure(self.blue_set)
            ).show(frame=False)
        else:
            raise ValueError(
                "Unknown mode for displaying the coloring, mode must be one of net, graph and polyhedron."
            )

    def print_Chimera_commands(self, interline: str="\n") -> None:
        """Print the Chimera UCSF commands that generate the corresponding oligomer."""
        alphabet = _vertices_to_dimers()
        blue_letters = chain.from_iterable(alphabet[v] for v in self.blue_set)
        red_letters = chain.from_iterable(alphabet[v].upper() for v in self.red_set)
        if self.blue_set:
            print(f"sel #1:.{':.'.join(blue_letters)}")
        if self.red_set:
            print(f"sel #2:.{':.'.join(red_letters)}")
        if interline:
            print(interline)


def unique_colorings(
    nb_blue_vertices=6,
    *,
    default_graph=True,
    graph: Graph | None = None,
    isomorphism=True,
) -> Iterable[Bicoloring]:
    """List the colorings with a given number of blue vertices in the directed graph,
    either up to rotations (if isomorphism is True) or not. Return either a list or a set.

    Keyword arguments:
    nb_blue_vertices -- number of vertices to be colored in blue
    default_graph -- whether to use the directed octahedral graph
    graph -- graph to be colored if default_graph is False
    isomorphism -- if True, the colorings are counted up to color-preserving automorphisms
    of the graph. If False, all colorings are listed.
    """

    if default_graph:
        graph = directed_cuboctahedral_graph()
    elif graph is None:
        raise ValueError("Set default_graph to True or provide graph.")

    assert 0 <= nb_blue_vertices <= graph.order()
    return (set if isomorphism else list)(
        OctahedralBicoloring(blue_set) if default_graph else Bicoloring(graph, blue_set)
        for blue_set in combinations(graph.vertices(), nb_blue_vertices)
    )


def write_to_csv(
    colorings: Iterable[Bicoloring],
    csv_file: str | None = None,
    *,
    csv_header=True,
    dialect="excel",
):
    """Write the contents of colorings to csv_file.

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
        for coloring in colorings:
            writer.writerow(
                [
                    len(coloring.blue_set),
                    len(coloring.red_set),
                    coloring.adjacencies.BR,
                    coloring.adjacencies.BB,
                    coloring.adjacencies.RR,
                    coloring.adjacencies.RB,
                    coloring.automorphism_group.order(),
                ]
            )

    if csv_file:
        with open(csv_file, "a") as file:
            write(file)
    else:
        write(sys.stdout)


def short_display(
    nb_blue_vertices: int,
    csv_=False,
    graph: Graph = directed_cuboctahedral_graph(),
    **csv_options,
):
    """Display the unique colorings for a given number of blue vertices.

    Keyword arguments:
    csv_ -- whether to output in the csv format
    csv_options -- options passed to write_to_csv if csv_ is True
    """
    colorings = unique_colorings(
        nb_blue_vertices=nb_blue_vertices, graph=graph, **options
    )

    if csv_:
        write_to_csv(colorings, **csv_options)
    else:
        colorings = sorted(colorings, key=lambda c: c.adjacencies.BR, reverse=True)
        lines = Counter(str(c) for c in colorings)
        print(
            f"With {nb_blue_vertices} blue vertices and {graph.order() - nb_blue_vertices} orange vertices,"
            f" the number of distinct arrangements is {len(colorings)}."
        )
        for val, key in lines.items():
            print(val + f"\t({key} such arrangements)" if key > 1 else val)


def overlap() -> dict[Bicoloring]:
    """List the pairs (c1, c2) where c1 has 6 blue vertices, c2 has 7 blue
    vertices and c2 is obtained from c1 by changing a single vertex color.
    """
    colorings1 = (
        c for c in unique_colorings(6, isomorphism=False) if c.adjacencies.BR == 8
    )
    colorings2 = [
        c for c in unique_colorings(7, isomorphism=False) if c.adjacencies.BR == 8
    ]
    return {c1: {c2 for c2 in colorings2 if c1.distance(c2) == 1} for c1 in colorings1}


def _experimental_coloring(switch=True) -> Bicoloring:
    """Return the coloring which the data seem to indicate, with the last vertex either
    red or blue as switch is True or False."""
    return OctahedralBicoloring(
        blue_set=[10, 2, 1, 11, 4, 5, 7] + ([] if switch else [0])
    )


if __name__ == "__main__":
    for c in unique_colorings(6):
        c.show(mode="net")
        c.show(mode="graph")
        c.show(mode="polyhedron")
