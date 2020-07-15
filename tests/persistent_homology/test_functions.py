"""Unit tests for persistent homology of functions."""

import unittest

from pyper.persistent_homology import calculate_persistence_diagrams_1d


class TestBeketayev(unittest.TestCase):
    def test(self):
        for order in ['sublevel', 'superlevel']:

            D = calculate_persistence_diagrams_1d(
                [3, 1, 6, 5, 8, 2, 7, 4],
                order=order
            )

            E = calculate_persistence_diagrams_1d(
                [3, 1, 8, 2, 7, 5, 6, 4],
                order=order
            )

            D = [(c, d) for c, d in D]
            E = [(c, d) for c, d in E]

            assert D == E
