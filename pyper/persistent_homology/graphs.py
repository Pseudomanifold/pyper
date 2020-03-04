"""Filtrations and persistent homology calculations for graphs."""

import igraph as ig
import numpy as np

from ..utilities import UnionFind
from ..representations import PersistenceDiagram


def _check_vertex_attribute_existence(graph, attribute_in):
    return attribute_in in graph.vs.attributes()


def _check_dimensionality(graph, attribute_in, x):
    attributes = graph.vs[attribute_in]

    # Ensure that both the attribute and the coordinate vectors have the
    # same dimensionality.
    a_lengths = set([len(attrib) for attrib in attributes])
    x_length = len(x)

    assert len(a_lengths) == 1    # only a single value
    assert x_length in a_lengths  # length/dimensionality coincide


def calculate_distance_filtration(
    graph,
    order=2,
    attribute_in='position',
    attribute_out='f'
):
    """Calculate a standard distance-based filtration for a given graph.

    Calculates a standard distance-based filtration for a given graph,
    i.e. a filtration that evaluates the Euclidean distance between
    nodes and uses this as the weight of an edge. Vertex weights will
    be set to zero, which is in line with a distance filtration.

    Parameters
    ----------
    graph:
        Input graph

    order:
        Order of the distance calculation. By default, the Euclidean
        distance between node attributes will be calculated.

    attribute_in:
        Specifies the (vertex) attribute that contains the coordinate
        data. This is assumed to be a high-dimensional vector, but the
        function will accept *any* existing attribute.

    attribute_out:
        Specifies the attribute name for storing the result of the
        calculation. This name will pertain to *both* vertices and
        edges.

    Returns
    -------
    Copy of the input graph, with vertex weights and each weights added
    as attributes `attribute_out`, respectively.
    """
    # Let's  make a copy first because we are modifying the graph's
    # attributes in place here.
    graph = ig.Graph.copy(graph)

    assert _check_vertex_attribute_existence(graph, attribute_in)

    vertex_weights = []
    edge_weights = []

    for edge in graph.es:

        u, v = edge.source, edge.target

        p = graph.vs[u][attribute_in]
        q = graph.vs[v][attribute_in]

        # Normal vector space distance of a given order; by default,
        # this will be the Euclidean distance.
        d = np.linalg.norm(p - q, ord=order)

        # This is a simple way to simulate a distance-based
        # filtration.
        vertex_weights.append(0.0)
        edge_weights.append(d)

    graph.vs[attribute_out] = vertex_weights
    graph.es[attribute_out] = edge_weights

    return graph


def calculate_height_filtration(
    graph,
    direction,
    attribute_in='position',
    attribute_out='f',
):
    """Calculate height filtration of a graph in some direction.

    *Note*: This function works for *all* vector-valued attributes of
    a graph, but in the following, it will be assumed that those
    attributes are 3D.

    Given a 3D direction vector, this function calculates a height
    filtration. To this end, a predefined vertex attribute will be
    evaluated. The attribute needs to contain 3D data. The result,
    i.e. the dot product between the attribute and direction, will
    be stored in an output attribute.

    The implementation follows the description of the *Persistent
    Homology Transform* [1].

    Parameters
    ----------
    graph:
        Input graph. Needs to contain a 3D node attribute that can be
        queried for the calculation. See `attribute_in` to change the
        name of the attribute.

    attribute_in:
        Specifies the (vertex) attribute that contains the 3D data.

    attribute_out:
        Specifies the attribute name for storing the result of the
        calculation. This name will pertain to *both* vertices and
        edges.

    Returns
    -------
    Copy of the input graph, with vertex weights and each weights added
    as attributes with the name `attribute_out`.

    References
    ----------
        [1]: Katharine Turner, Sayan Mukherjee, Doug M Boyer: "Persistent
        Homology Transform for Modeling Shapes and Surfaces",
        arXiv:1310.1030.
    """
    assert _check_vertex_attribute_existence(graph, attribute_in)
    assert _check_dimensionality(graph, attribute_in, direction)

    # Let's  make a copy first because we are modifying the graph's
    # attributes in place here.
    graph = ig.Graph.copy(graph)

    # Following the original terminology in the paper
    v = direction

    for vertex in graph.vs:
        x = vertex[attribute_in]
        r = np.dot(x, v)

        # TODO: check whether attribute does not yet exist
        vertex[attribute_out] = r

    for edge in graph.es:
        source, target = graph.vs[edge.source], graph.vs[edge.target]

        # The original paper describes a sublevel set filtration, so it
        # is sufficient to use the `max` function here.
        r = max(source[attribute_out], target[attribute_out])
        edge[attribute_out] = r

    return graph


def calculate_persistence_diagrams(
    graph,
    vertex_attribute='f',
    edge_attribute='f',
    order='sublevel',
):
    """Calculate persistence diagrams for a graph.

    Calculates a set of persistence diagrams for a given graph. The
    graph is already assumed to contain function values on its edge
    and node elements, respectively. Based on this information, the
    function will calculate persistence diagrams using sublevel, or
    superlevel, sets.

    Parameters
    ----------
    graph:
        Input graph. Needs to have vertex and edge attributes for the
        calculation to be valid.

    vertex_attribute:
        Specifies which vertex attribute to use for the calculation of
        persistent homology.

    edge_attribute:
        Specifies with edge attribute to use for the calculation of
        persistent homology.

    order:
        Specifies the filtration order that is to be used for calculating
        persistence diagrams. Can be either 'sublevel' for a sublevel set
        filtration, or 'superlevel' for a superlevel set filtration.

    Returns
    -------
    Set of persistence diagrams, describing topological features of
    a specified dimension.
    """
    n_vertices = graph.vcount()
    uf = UnionFind(n_vertices)

    # TODO: check for vertex attributes
    # TODO: check for edge attributes

    # The edge weights will be sorted according to the pre-defined
    # filtration that has been specified by the client.
    edge_weights = np.array(graph.es[edge_attribute])
    edge_indices = None

    # Will contain all the edges that are responsible for cycles in the
    # graph.
    edge_indices_cycles = []

    assert order in ['sublevel', 'superlevel']

    if order == 'sublevel':
        edge_indices = np.argsort(edge_weights, kind='stable')

    # Like the professional that I am, we just need to flip the edge
    # weights here. Note that we do not make *any* assumptions about
    # whether this is consistent with respect to the nodes.
    else:
        edge_indices = np.argsort(-edge_weights, kind='stable')

    # Will be filled during the iteration below. This will become
    # the return value of the function.
    persistence_diagram_0 = PersistenceDiagram()

    # Go over all edges and optionally create new points for the
    # persistence diagram.
    for edge_index, edge_weight in \
            zip(edge_indices, edge_weights[edge_indices]):
        u, v = graph.es[edge_index].tuple

        # Preliminary assignment of younger and older component. We
        # will check below whether this is actually correct, for it
        # is possible that u is actually the older one.
        younger = uf.find(u)
        older = uf.find(v)

        # Nothing to do here: the two components are already the
        # same
        if younger == older:
            edge_indices_cycles.append(edge_index)
            continue

        # Ensures that the older component precedes the younger one
        # in terms of its vertex index
        elif younger > older:
            u, v = v, u
            younger, older = older, younger

        vertex_weight = graph.vs[vertex_attribute][younger]

        creation = vertex_weight    # x coordinate for persistence diagram
        destruction = edge_weight   # y coordinate for persistence diagram

        uf.merge(u, v)
        persistence_diagram_0.add(creation, destruction)

    # By default, use the largest weight to assign to unpaired
    # vertices. This is consistent with *extended persistence*
    # calculations.
    unpaired_value = edge_weights[edge_indices[-1]]

    # Add tuples for every root component in the Union--Find data
    # structure. This ensures that multiple connected components
    # are handled correctly.
    for root in uf.roots():

        vertex_weight = graph.vs[vertex_attribute][root]

        creation = vertex_weight
        destruction = unpaired_value

        persistence_diagram_0.add(creation, destruction)

    # Create a persistence diagram for the cycles in the data set.
    # Notice that these are *not* properly destroyed; a better, or
    # smarter, calculation would be warranted.

    persistence_diagram_1 = PersistenceDiagram()

    for edge_index in edge_indices_cycles:
        persistence_diagram_1.add(edge_weights[edge_index], unpaired_value)

    return persistence_diagram_0, persistence_diagram_1
