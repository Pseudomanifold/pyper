"""Persistence diagram class and functions.

Contains a class describing persistence diagrams, along with some basic
summary statistics.
"""

import collections


class PersistenceDiagram(collections.Sequence):
    """Persistence diagram class.

    Represents a persistence diagram, i.e. a pairing of nodes in
    a graph. The purpose of this class is to provide a *simpler*
    interface for storing and accessing this pairing.
    """

    def __init__(self):
        """Create new persistence diagram."""
        self._pairs = []

    def __len__(self):
        """Return the number of pairs in the persistence diagram."""
        return len(self._pairs)

    def __getitem__(self, index):
        """Return the persistence pair at the given index."""
        return self._pairs[index]

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

        Parameters
        ----------
        p:
            Exponent for the total persistence calculation

        Returns
        -------
        Total persistence with exponent $p$.
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
