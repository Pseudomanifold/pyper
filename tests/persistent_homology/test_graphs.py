"""Unit tests for persistent homology of graphs."""

import unittest

import igraph as ig

from pyper.persistent_homology import calculate_persistence_diagrams


def _make_test_graph_gfl():
    """Create test graph (GFL).

    Creates a vertex-valued test graph from the graph filtration
    learning paper [1].

    References
    ----------
        [1]: Christoph D. Hofer, Florian Graf, Bastian Rieck, Marc
        Niethammer, Roland Kwitt: "Graph Filtration Learning",
        arXiv:1905.10996.
    """

    G = ig.Graph()
    G.add_vertices(5)

    G.add_edges([
        (0, 1), (1, 2), (2, 3), (2, 4)
    ])

    G.vs['f'] = [1, 4, 3, 2, 1]
    G.es['f'] = [4, 4, 3, 3]

    return G


class TestGFLGraph(unittest.TestCase):
    def test(self):
        G = _make_test_graph_gfl()

        pd_0, pd_1 = calculate_persistence_diagrams(G)

        self.assertIn((1, 4), pd_0)
        self.assertIn((2, 3), pd_0)

        # TODO: check existence of other tuples
