"""Persistence diagram class and functions.

Contains a class describing persistence diagrams, along with some basic
summary statistics.
"""

import collections.abc
import numpy


class PersistenceDiagram(collections.abc.Sequence):
    """Persistence diagram class.

    Represents a persistence diagram, i.e. a pairing of nodes in
    a graph. The purpose of this class is to provide a *simpler*
    interface for storing and accessing this pairing.
    """

    def __init__(self, pairs=None, dimension=None):
        """Create new persistence diagram.

        Parameters
        ----------
        pairs:
            Optional sequence of persistence pairs to add to this
            diagram. If set, the diagram will be initialised with
            said sequence. The sequence *must* consist of tuples.

        dimension : int (optional)
            Specifies dimension of the diagram. This can be useful in
            order to denote the tuples that gave rise to a *specific*
            diagram.
        """
        if pairs is None:
            self._pairs = []
        else:
            self._pairs = [(c, d) for c, d in pairs]

        self._dimension = dimension

    def __len__(self):
        """Return the number of pairs in the persistence diagram."""
        return len(self._pairs)

    def __getitem__(self, index):
        """Return the persistence pair at the given index."""
        return self._pairs[index]

    def __truediv__(self, alpha):
        """Elementwise division by a scalar."""
        self._pairs = [(c / alpha, d / alpha) for c, d in self._pairs]
        return self

    @property
    def dimension(self):
        """Return dimension of the persistence diagram, if set."""
        return self._dimension

    @property
    def persistence(self):
        """Return list of persistence values of this diagram."""
        return [abs(d - c) for c, d in self._pairs]

    def add(self, x, y):
        """Append a new persistence pair to the given diagram.

        Extends the persistence diagram by adding a new persistence
        tuple to the diagram. Performs no other validity checks.

        Parameters
        ----------
        x:
            Creation value of the given persistence pair
        y:
            Destruction value of the given persistence pair
        """
        self._pairs.append((x, y))

    def union(self, other):
        """Calculate the union of two persistence diagrams.

        The union of two persistence diagrams is defined as the union of
        their underlying persistence pairs. The current persistence diagram
        is modified in place.

        Parameters
        ----------
        other:
            Other persistence diagram

        Returns
        -------
        Updated persistence diagram.
        """
        for x, y in other:
            self.add(x, y)

        return self

    def total_persistence(self, p=1):
        """Calculate the total persistence of the current pairing.

        The total persistence is closely related to the $p$-norm in that
        it employs a sum of the persistence values found in the diagram.
        In contrast to the norm, though, no additional root will be used
        in this formulation.

        Parameters
        ----------
        p:
            Exponent for the total persistence calculation

        Returns
        -------
        Total persistence with exponent $p$.
        """
        return sum([abs(x - y)**p for x, y in self._pairs])

    def p_norm(self, p=1):
        """Calculate the $p$-norm of the current pairing.

        Parameters
        ----------
        p : float
            Exponent for the $p$-norm calculation

        Returns
        -------
        $p$-norm of the persistence diagram.
        """
        return sum([abs(x - y)**p for x, y in self._pairs])**(1.0 / p)

    def infinity_norm(self, p=1):
        """Calculate the infinity norm of the current pairing.

        Parameters
        ----------
        p:
            Exponent for the infinity norm calculation

        Returns
        -------
        Infinity norm with exponent $p$.
        """
        return max([abs(x - y)**p for x, y in self._pairs])


def diagrams_from_tensor(data):
    """Attempt to extract persistence diagrams from tensor.

    This function attempts to turn a tensor into a sequence of
    persistence diagrams. The tensor is supposed to consist of
    the axes creation, destruction, and dimension. Every other
    axis will be ignored.

    Parameters
    ----------
    data : np.array
        Input tensor of shape (n, 3). Each tuple represents a topological
        feature by its creation, destruction, and dimension.

    Returns
    -------
    List of persistence diagrams. The property `dimension` of each
    diagram is set to indicate the dimension of its tuples.
    """
    # Will contain the individual diagrams, sorted by dimension. This is
    # not necessary, but a nice touch.
    diagrams = []
    dimensions = numpy.unique(data[..., 2])

    for dimension in sorted(dimensions):
        data_ = data[data[..., 2] == dimension]
        persistence_pairs = data_[:, 0:2]

        diagram = PersistenceDiagram(persistence_pairs, dimension=dimension)
        diagrams.append(diagram)

    return diagrams
