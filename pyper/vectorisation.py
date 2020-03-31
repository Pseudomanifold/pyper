"""Feature vector creation approaches for persistence diagrams.

This module contains feature vector creation approaches for persistence
diagrams that permit the use in modern machine learning algorithms.
"""


def _persistence(x, y):
    """Auxiliary function for calculating persistence of a tuple."""
    return abs(x - y)


def featurise_distances(diagram):
    """Create feature vector by distance-to-diagonal calculation.

    Creates a feature vector by calculating distances to the diagonal
    for every point in the diagram and returning a sorted vector. The
    representation is *stable* but might not be discriminative.

    Parameters
    ----------
    diagram : `PersistenceDiagram`
        Persistence diagram to featurise. Can also be a generic 2D
        container for iterating over tuples.

    Returns
    -------
    Sorted vector of distances to diagonal. The vector is sorted in
    descending order, such that high persistence points precede the
    ones of low persistence.
    """
    distances = [_persistence(x, y) for x, y in diagram]
    return sorted(distances, reverse=True)


def featurise_pairwise_distances(diagram):
    """Create feature vector based on stable signatures.

    Creates a feature vector by calculating the minimum of pairwise
    distances and distances to the diagonal of each pair of points.
    This representation follows the paper:

        Mathieu Carrière, Steve Y. Oudot, and Maks Ovsjanikov
        Stable Topological Signatures for Points on 3D Shapes

    The representation is stable, but more costly to compute.

    Parameters
    ----------
    diagram : `PersistenceDiagram`
        Persistence diagram to featurise. Can also be a generic 2D
        container for iterating over tuples.

    Returns
    -------
    Sorted vector of distances as described above. The vector is sorted
    in *descending* order.
    """
    distances = []

    # Auxiliary function for calculating the infinity distance between
    # the two points.
    def _distance(a, b, x, y):
        return max(abs(a - x), abs(b - y))

    for i, (a, b) in enumerate(diagram):
        for j, (x, y) in enumerate(diagram[i:]):
            m = min(
                    _distance(a, b, x, y),
                    _persistence(a, b),
                    _persistence(x, y)
                )

            distances.append(m)

    return sorted(distances, reverse=True)
