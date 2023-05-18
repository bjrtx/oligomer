#!/usr/bin/env sage
# pylint: disable=import-error
"""The behaviour of this code is described in [companion paper].
It requires the SageMath library and can be run with SageMath.
See installation instructions at https://www.sagemath.org/
or use an online interpreter such as https://sagecell.sagemath.org/
"""

from __future__ import annotations
from itertools import chain, combinations
from functools import cache, cached_property
import collections
from collections.abc import Iterable, Collection
from dataclasses import dataclass, field

import csv
import sys

# Import explicitly to allow use as Python code
import sage
import sage.all
from sage.geometry.polyhedron.library import polytopes
from sage.geometry.polyhedron.constructor import Polyhedron


# Define colors matching UCSF Chimera's
_CHIMERA_RED = (1.000, 0.427, 0.000)
_MEDIUM_BLUE = sage.plot.colors.Color("#3232cd").rgb()

# Vertex size for plotting graphs
_VERTEX_SIZE = 400


# Define type aliases
Graph = sage.graphs.generic_graph.GenericGraph
DiGraph = sage.graphs.digraph.DiGraph


@cache
def bfr_graph() -> DiGraph:
    """Return the Bfr graph with a specific edge orientation,
    as described in the companion paper. Its vertices are labeled from
    0 to 11 inclusive. The graph cannot be modified.
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
    # Fix vertex positions for drawing the graph
    vertex_positions = dict(
        enumerate(
            [
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
        )
    )
    return DiGraph(
        out_neighbours, pos=vertex_positions, immutable=True, format="dict_of_lists"
    )


@cache
def _vertices_to_dimers() -> dict[int, str]:
    """Return a dict mapping each vertex label (integers from 0 to 12) of the directed
    Bfr graph to the two letters designing its dimers (first top then bottom) in
    Chimera-generated net pictures.
    """
    return {
        5: "bo",
        6: "aq",
        9: "lw",
        1: "is",
        2: "jt",
        8: "gk",
        10: "ep",
        0: "fr",
        3: "du",
        11: "mx",
        4: "cv",
        7: "hn",
    }


@cache
def _vertices_to_facets() -> dict[int, Polyhedron]:
    """Return a dict mapping each vertex of the directed Bfr graph to a facet of
    the rhombic dodecahedron. The facets are Polyhedron objects, which can be used for
    creating 3D models of the dodecahedron."""
    facets = [f.as_polyhedron() for f in polytopes.rhombic_dodecahedron().facets()]
    # The edges correspond to facets that intersect (along a line, a 1D object)
    edges = [
        (i, j)
        for (i, f), (j, g) in combinations(enumerate(facets), 2)
        if f.intersection(g).dimension() > 0
    ]
    assert len(edges) == 24 and len(facets) == 12
    # Find a graph isomorphism between the intersection graph defined by these
    # edges and the Bfr graph.
    _, isomorphism_map = sage.graphs.graph.Graph(data=edges).is_isomorphic(
        bfr_graph().to_undirected(), certificate=True
    )
    return {isomorphism_map[i]: f for i, f in enumerate(facets)}


def oligomer_structure(blue_set: Iterable[int] = frozenset()):
    """Return a Sage Graphics object representing a 24-mer whose facets are colored
    according to blue_set, that is either blue if their number is in blue_set or
    otherwise red.
    """
    # The facets correspond to dimers
    facets = _vertices_to_facets()
    graph = bfr_graph()
    graphics_object = 0

    for i, facet in facets.items():  # for each dimer
        # Build the two 'outgoing' edges of the dimer, and their two midpoints
        edges_out = (facet.intersection(facets[j]) for j in graph.neighbors_out(i))
        midpoints = [e.center() for e in edges_out]
        # Build the two 'ingoing' edges
        edges_in = [facet.intersection(facets[j]) for j in graph.neighbors_in(i)]
        # Add the two quadrilaterals corresponding to the dimer's chains
        # to the Graphics object.
        graphics_object += sum(
            Polyhedron(vertices=chain(edge.vertices(), midpoints)).plot(
                line={"color": "black", "thickness": 8},
                polygon=_MEDIUM_BLUE if i in blue_set else _CHIMERA_RED,
                online=True,
            )
            for edge in edges_in
        )

    return graphics_object


@cache
def more_complicated_graph(halve_edges: bool = True) -> DiGraph:
    """Return a digraph encoding dimer-dimer junctions for the case of heterodimers.
    It has 24 vertices (one by chain) and the edges represent T-junctions
    between chains. Note that two chains belonging to the same dimer are not
    adjacent in this graph."""
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
    # here a monomer is a tuple (k, i) where k is a dimer and i is either 0 or 1
    if halve_edges:
        out_neighbours = (((k, 0), v[0:1]) for k, v in out_neighbours.items())
    else:
        out_neighbours = (
            ((k, i), v) for k, v in out_neighbours.items() for i in (0, 1)
        )

    out_neighbours = dict(chain(out_neighbours))

    # Fix vertex positions for drawing the graph
    eps = 0.3
    vertex_positions = dict(
        enumerate(
            [
                ([-1 - eps, -1 - eps], [-1 + eps, -1 + eps]),
                ([-1 - eps, 1 + eps], [-1 + eps, 1 - eps]),
                ([1 - eps, 1 - eps], [1 + eps, 1 + eps]),
                ([1 - eps, -1 + eps], [1 + eps, -1 - eps]),
                ([-5 - eps, -5 - eps], [-5 + eps, -5 + eps]),
                ([-5 + eps, 5 - eps], [-5 - eps, 5 + eps]),
                ([5 - eps, 5 - eps], [5 + eps, 5 + eps]),
                ([5 + eps, -5 - eps], [5 - eps, -5 + eps]),
                ([-3 + eps, 0], [-3 - 2 * eps, 0]),
                ([0, 3 + 2 * eps], [0, 3 - eps]),
                ([3 + 2 * eps, 0], [3 - eps, 0]),
                ([0, -3 + eps], [0, -3 - 2 * eps]),
            ]
        )
    )
    vertex_positions = {
        (k, i): pos[i] for k, pos in vertex_positions.items() for i in (0, 1)
    }
    return DiGraph(out_neighbours, pos=vertex_positions, immutable=True)


# Several mathematical assertions concerning this directed graph
if __name__ == "__main__":
    dg = bfr_graph()
    # Up to forgetting edge directions, it is isomorphic to the Cayley graph of
    # an alternating group
    assert dg.is_isomorphic(
        sage.all.AlternatingGroup(4).cayley_graph(), edge_labels=False
    )
    assert dg.to_undirected().is_isomorphic(sage.all.polytopes.cuboctahedron().graph())
    # The automorphism group of dg is (isomorphic to) S_4 and can be interpreted as the
    # rotational octahedral symmetry group (O, 432, etc.).
    assert sage.all.SymmetricGroup(4).is_isomorphic(dg.automorphism_group())

# An object for storing adjacency information.
Adjacencies = collections.namedtuple("Adjacencies", ["BB", "RR", "BR", "RB"])
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
    blue_set: Iterable = frozenset()
    red_set: Iterable = field(init=False)

    def __post_init__(self):
        # using __setattr__ because the class if frozen (its values cannot be modified
        # in the standard way)
        if not self.graph.is_immutable():
            object.__setattr__(self, "graph", self.graph.copy(immutable=True))
        object.__setattr__(self, "blue_set", frozenset(self.blue_set))
        vertices = frozenset(self.graph.vertices())
        object.__setattr__(self, "red_set", vertices - self.blue_set)

    @cached_property
    def canonical_form(self) -> Bicoloring:
        """Return an object that characterises the coloring up to automorphisms.
        In practice, this is a relabelling of self.graph in which the vertices of
        self.blue_set come first. This relabelling is the same for two colorings of
        the same graph with a color-preserving isomorphism.
        """
        # canon is self.graph relabeled and mapping is a dictionary describing the
        # relabelling
        canon, mapping = self.graph.canonical_label(
            partition=[self.blue_set, self.red_set], certificate=True
        )
        return Bicoloring(canon, blue_set={mapping[v] for v in self.blue_set})

    def nb_adjacencies(self, left: Iterable, right: Iterable) -> int:
        """Count the edges from the set left to the set right."""
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

    def __eq__(self, other: Bicoloring) -> bool:
        """Tests for isomorphism with respect to the color classes."""
        return self.canonical_form.graph == other.canonical_form.graph

    def __hash__(self):
        return hash(self.canonical_form.graph)

    def distance(self, other: Bicoloring) -> int:
        """Count the vertices which have distinct colors in self and in other."""
        return len(self.blue_set.symmetric_difference(other.blue_set))

    def show(self, mode="graph") -> None:
        """Displays a picture of the coloring."""
        if mode is not None and mode != "graph":
            raise ValueError("Unexpected mode in Bicoloring.show")
        self.graph.plot(
            vertex_size=_VERTEX_SIZE,
            vertex_color=_CHIMERA_RED,
            vertex_labels = _vertices_to_dimers(), # None
            vertex_colors={_MEDIUM_BLUE: self.blue_set},
        ).show(transparent=True)


@dataclass(frozen=True, eq=False)  # inherit inequality from the parent class
class BfrBicoloring(Bicoloring):
    """Provide methods specific to bicolorings of the directed Bfr graph."""

    def __init__(self, blue_set: Iterable[int] = frozenset()):
        super().__init__(graph=bfr_graph(), blue_set=blue_set)

    def show(self, mode="net") -> None:
        """Displays a picture of the coloring, either as a net (flat map),
        as a graph or as a 3D polyhedron.

        Keyword arguments:
        mode -- one of "net", "graph", "polyhedron"
        """
        if mode == "net":
            # Build a diamond shape for each dimer
            def diamond(center, idx):
                x, y = center
                alpha = 0.95  # sligthly less than 1 for spacing
                return (
                    sage.all.polygon(
                        [
                            (x - alpha, y),
                            (x, y - alpha),
                            (x + alpha, y),
                            (x, y + alpha),
                        ],
                        color=(_MEDIUM_BLUE if idx in self.blue_set else _CHIMERA_RED),
                        edgecolor="black",
                    )
                    + sage.all.line(
                        [(x - alpha / 2, y - alpha / 2), (x + alpha / 2, y + alpha / 2)]
                        if y == 1
                        else [
                            (x - alpha / 2, y + alpha / 2),
                            (x + alpha / 2, y - alpha / 2),
                        ],
                        rgbcolor="black",
                    )
                    + sage.all.text(_vertices_to_dimers()[idx], center, color="black")
                )

            centers = {
                5: (7, 2),
                6: (6, 1),
                9: (5, 2),
                1: (3, 2),
                2: (4, 1),
                8: (1, 2),
                10: (5, 0),
                0: (2, 1),
                3: (3, 0),
                11: (1, 0),
                4: (0, 1),
                7: (7, 0),
            }

            sum(diamond(center, idx) for idx, center in centers.items()).show(
                axes=False, transparent=True
            )

        elif mode == "graph":
            super().show()
        elif mode == "polyhedron":
            oligomer_structure(self.blue_set).show(frame=False, transparent=True)
        else:
            raise ValueError(
                "Unknown mode for displaying the coloring: mode must be one of net,"
                "graph and polyhedron."
            )

    # pylint: disable-next=invalid-name
    def print_Chimera_commands(self, end: str = "\n", file=sys.stdout) -> None:
        """Print the Chimera UCSF commands that generate the corresponding oligomer."""
        blue_letters = chain.from_iterable(
            _vertices_to_dimers()[v] for v in self.blue_set
        )
        red_letters = chain.from_iterable(
            _vertices_to_dimers()[v].upper() for v in self.red_set
        )
        if self.blue_set:
            print(f"sel #1:.{':.'.join(blue_letters)}", file=file)
        if self.red_set:
            print(f"sel #2:.{':.'.join(red_letters)}", file=file)
        if end:
            print(end, file=file)


def unique_colorings(
    nb_blue_vertices: int,
    *,
    default_graph=True,
    graph: Graph | None = None,
    isomorphism=True,
) -> Collection[Bicoloring]:
    """List the colorings with a given number of blue vertices in the directed graph,
    either up to rotations (if isomorphism is True) or not. Return a list or a set.

    Keyword arguments:
    nb_blue_vertices -- number of vertices to be colored in blue
    default_graph -- whether to use the directed Bfr graph
    graph -- graph to be colored if default_graph is False
    isomorphism -- if True, colorings are counted up to color-preserving automorphisms
    of the graph. If False, all colorings are listed.
    """

    if default_graph:
        graph = bfr_graph()
    elif graph is None:
        raise ValueError("Set default_graph to True or provide graph.")

    assert 0 <= nb_blue_vertices <= graph.order()
    return (set if isomorphism else list)(
        BfrBicoloring(blue_set) if default_graph else Bicoloring(graph, blue_set)
        for blue_set in combinations(graph.vertices(), nb_blue_vertices)
    )


def write_to_csv(
    colorings: Iterable[Bicoloring],
    csv_file: str | None = None,
    *,
    csv_header=True,
    dialect="excel",
):
    """Write the contents of colorings in the CSV format to csv_file.

    Keyword arguments:
    csv_file -- optional: a path-like object (such as a filename string)
    If None, the output will be written to the standard output.
    csv_header -- if True, add a header with column names as the first row
    dialect -- specific CSV format, see Python's csv module documentation
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
        with open(csv_file, "a", encoding="locale") as file:
            write(file)
    else:
        write(sys.stdout)


def short_display(
    nb_blue_vertices: int,
    csv_=False,
    graph: Graph | None = None,
    **csv_options,
):
    """Display the unique colorings for a given number of blue vertices.

    Keyword arguments:
    csv_ -- whether to output in the csv format
    csv_options -- options passed to write_to_csv if csv_ is True
    """
    colorings = unique_colorings(
        nb_blue_vertices=nb_blue_vertices, graph=graph, **csv_options
    )
    graph = graph or bfr_graph()
    if csv_:
        write_to_csv(colorings, **csv_options)
    else:
        # sort the colorings by decreasing number of blue->red adjacencies
        colorings = sorted(colorings, key=lambda c: c.adjacencies.BR, reverse=True)
        descriptions = collections.Counter(str(c) for c in colorings)
        print(
            f"With {nb_blue_vertices} blue vertices and "
            f"{graph.order() - nb_blue_vertices} orange vertices, the number of "
            f"distinct arrangements is {len(colorings)}."
        )
        for desc, count in descriptions.items():
            print(desc + f"\t({count} such arrangements)" if desc > 1 else count)


def overlap() -> dict[Bicoloring, set[Bicoloring]]:
    """Return a dictionary whose value are sets. The Bicoloring c2 appears in the
    set value associated with c1 if and only if c1 has 6 blue vertices, c2 has 7 blue
    vertices and c2 is obtained from c1 by changing a single vertex color.
    """
    colorings6 = (
        c for c in unique_colorings(6, isomorphism=False) if c.adjacencies.BR == 8
    )
    colorings7 = [
        c for c in unique_colorings(7, isomorphism=False) if c.adjacencies.BR == 8
    ]
    return {c1: {c2 for c2 in colorings7 if c1.distance(c2) == 1} for c1 in colorings6}


def _experimental_coloring(switch=True) -> BfrBicoloring:
    """Return the coloring which the data seem to indicate, with the last vertex either
    red or blue as switch is True or False."""
    return BfrBicoloring(blue_set=[10, 2, 1, 11, 4, 5] + ([] if switch else [0]))


_experimental_coloring().show("graph")
_experimental_coloring().show("net")
_experimental_coloring().print_Chimera_commands()
