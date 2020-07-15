"""Filtrations and persistent homology calculation for functions."""

import enum
import operator

import numpy as np

from ..utilities import UnionFind
from ..representations import PersistenceDiagram


def calculate_persistence_diagrams_1d(
    function,
    order='sublevel',
):
    """Calculate persistence diagrams for a 1D function.

    Calculates persistence diagrams for a 1D function, following the
    usual Morse filtration. This is equivalent to calculating one of
    the "merge" or "split" trees.

    Parameters
    ----------
    function:
        Input function. Should be an array or an array-like data
        structure that affords iteration.

    order:
        Specifies the filtration order that is to be used for calculating
        persistence diagrams. Can be either 'sublevel' for a sublevel set
        filtration, or 'superlevel' for a superlevel set filtration.

    Returns
    -------
    Persistence diagram of the merge or split tree.
    """
    assert order in ['sublevel', 'superlevel']

    function = np.asarray(function)

    if order == 'sublevel':
        indices = np.argsort(function, kind='stable')
        predicate = operator.lt
    else:
        indices = np.argsort(-function, kind='stable')
        predicate = operator.gt

    n = len(function)

    # Union--Find data structure for tracking the indices of the
    # persistence diagram.
    uf = UnionFind(n)

    # Will contain persistence pairs as index tuples where each
    # index refers to a point in the input series.
    persistence_pairs = []

    # In the comments of this function, I will assume that `predicate`
    # refers to the 'less than' operator. Hence, I will speak of local
    # minima and so on. This is done to make the code more readable.
    for index in indices:
        x = function[index]
        u = x
        v = x

        # Grab neighbours, if available. Else, we just pretend that we
        # discovered x again. Since we do not check for equality below
        # this works out just fine.
        if index > 0:
            u = function[index - 1]

        if index < n - 1:
            v = function[index + 1]

        # Case 1 [local minimum]: both neighbours have higher function
        # values
        if predicate(x, u) and predicate(x, v):
            # Nothing to do here
            pass

        # Case 2 [local maximum]: both neighbours have lower function
        # values
        elif predicate(u, x) and predicate(v, x):

            # For the persistence pairing, the 'older' branch persists,
            # while the 'younger' branch is being merged. To decide the
            # age, we use the predicate between both neighbours. Notice
            # that this also decides the 'direction' of the merge. It's
            # crucial to look up the lowest point in each component for
            # merging in the proper direction.
            if predicate(function[uf.find(index - 1)],
                         function[uf.find(index + 1)]):

                # u is the 'older' branch and persists; merge everything
                # into it.
                persistence_pairs.append((uf.find(index + 1), index))
                uf.merge(index, index + 1)
                uf.merge(index + 1, index - 1)
            else:

                # v is the 'older' branch and persists; merge everything
                # into it.
                persistence_pairs.append((uf.find(index - 1), index))
                uf.merge(index, index - 1)
                uf.merge(index - 1, index + 1)

        # Case 3 [regular point]: one neighbour has a higher function
        # value, the other one has a lower function value.
        else:

            # Indicates whether a merge should be done, and if so, which
            # direction should be used. `LEFT` refers to the left vertex
            # with respect to the current vertex, for example.
            class Merge(enum.Enum):
                LEFT, RIGHT = range(2)

            # Indicates all merges that need to be done. It is possible
            # that we want to merge both to the left and to the right.
            merges = []

            # Only add the edge to the *lower* point because we have
            # already encountered the point in filtration order. The
            # higher point will then add the second edge.
            if predicate(u, x):
                merges.append(Merge.LEFT)
            elif predicate(v, x):
                merges.append(Merge.RIGHT)

            # Check whether the point is incomparable with its left and
            # right neighbours, respectively. This will decide how they
            # can be merged.
            incomparable_l_nb = not predicate(u, x) and not predicate(x, u)
            incomparable_r_nb = not predicate(v, x) and not predicate(x, v)

            # At this point, we have already checked whether there are
            # points that are truly *lower* than the current point and
            # we have recorded their merges. In addition, we now check
            # for vertices for which the predicate yields incomparable
            # results, for this indicates that they are *equal* to the
            # current point, from the perspective of the predicate.
            #
            # Left neighbour is incomparable with current point, hence
            # we can choose it to create a new edge.
            if incomparable_l_nb:
                merges.append(Merge.LEFT)

            # Right neighbour is incomparable with current point, hence
            # we can choose it to create a new edge.
            if incomparable_r_nb:
                merges.append(Merge.RIGHT)

            # Depending on the merge direction, adjust all required
            # data structures. If we do not have at least one valid
            # merge, we just skip everything.
            if not merges:
                continue

            if Merge.LEFT in merges and index != 0:

                # Switch the order in which the merges happens if the
                # neighbours are incomparable. This is done because a
                # representative of a component should be the highest
                # index with respect to the ordering of the series so
                # that lower indices will become children of it.
                if incomparable_l_nb:
                    uf.merge(index - 1, index)
                else:
                    uf.merge(index, index - 1)

            if Merge.RIGHT in merges and index != n - 1:

                if incomparable_r_nb:
                    uf.merge(index + 1, index)
                else:
                    uf.merge(index, index + 1)

    # Merge the minimum with the maximum for the time series. This
    # ensures that the diagram always contains at least a *single*
    # persistence pair.
    if len(indices) >= 2:
        if (indices[0], indices[-1]) not in persistence_pairs:
            persistence_pairs.append((indices[0], indices[-1]))

    pd = PersistenceDiagram(
        [(function[c], function[d]) for c, d in persistence_pairs]
    )

    return pd
