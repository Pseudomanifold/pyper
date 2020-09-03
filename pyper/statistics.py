"""Statistics calculations for persistence diagrams.

This module contains additional summary statistics or auxiliary
functions that are useful when doing statistics with persistence
diagrams.
"""

import numpy as np

from scipy.spatial import Voronoi
from sklearn.neighbors import NearestNeighbors


def _get_persistence_values(diagram):
    """Auxiliary function for calculating persistence values."""
    return [abs(x - y) for x, y in diagram]


def _get_knn_distances(diagram, k=1):
    """Return distance to $k$ nearest neighbours."""
    # We follow the Chebyshev distance here because it is the right
    # geometry within the persistence diagram.
    nn = NearestNeighbors(n_neighbors=k, metric='chebyshev')
    nn.fit(diagram._pairs)

    # We are only interested in the distances!
    distances, _ = nn.kneighbors()
    return distances.ravel()


def _transform_pairs(diagram):
    """Return pairs transformed into creation--persistence plane."""
    pairs = [
        (a, abs(b - a)) for a, b in diagram._pairs
    ]

    return pairs


def persistent_entropy(diagram):
    """Calculate persistent entropy of a diagram.

    The persistent entropy is a simple measure of how different the
    persistence values of the diagram are. It was originally described
    by Rucco et al. [1].

    .. [1] Matteo Rucco, Filippo Castiglione, Emanuela Merelli, and
    Marco Pettini: *Characterisation of the Idiotypic Immune Network
    Through Persistent Entropy*.

    Parameters
    ----------
    diagram : PersistenceDiagram
        Persistence diagram whose persistent entropy should be
        calculated.

    Returns
    -------
    Persistent entropy of the input diagram as a single np.float64.
    """
    pers = _get_persistence_values(diagram)
    total_pers = np.sum(pers)
    probabilities = np.asarray([p / total_pers for p in pers])

    return np.sum(-probabilities * np.log2(probabilities))


def spatial_entropy_knn(diagram):
    """Calculate spatial entropy based on $k$ nearest neighbours.

    Calculates a simple spatial entropy of the diagram that is based
    on the *relative* distribution of points in the diagram.
    """
    distances = _get_knn_distances(diagram)

    # Approximate all 'cells' induced by the distance to the nearest
    # neighbour. The areas are circles, but in the Chebyshev metric,
    # the circle is actually a cube of side length $2d$, for $d$ the
    # distance to the nearest neighbour.
    areas = 4 * distances**2
    total_area = np.sum(areas)
    probabilities = np.array([areas / total_area for area in areas])

    # Ensures that a probability of zero will just result in
    # a logarithm of zero as well. This is required whenever
    # one deals with entropy calculations.
    log_prob = np.log2(probabilities,
                       out=np.zeros_like(probabilities),
                       where=(probabilities > 0))

    return np.sum(-probabilities * log_prob)


def spatial_entropy_voronoi(diagram):
    """Calculate spatial entropy based on Voronoi diagrams.

    Calculates a spatial entropy of the diagram that is based on the
    *relative* distribution of points in the diagram. This function,
    in contrast to `spatial_entropy_knn`, employs a Voronoi diagram.
    """

    points = np.asarray(_transform_pairs(diagram))

    # Add boundary vertices to ensure that the regions are bounded. This
    # is somewhat artificial, and I am not sure whether 'tis a good idea
    # to do so.

    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

    points = np.append(points, [
          (x_min, y_min),  # lower left
          (x_max, y_min),  # lower right
          (x_max, y_max),  # upper right
          (x_min, y_max),  # upper left
        ],
        axis=0
    )

    voronoi_diagram = Voronoi(points, qhull_options='Qbb Qc Qz')
    raise NotImplementedError('This function is not yet implemented')
