"""Unit tests for summary statistics calculation."""

import unittest

import numpy as np

from pyper.representations import PersistenceDiagram
from pyper.statistics import persistent_entropy


class TestPersistentEntropy(unittest.TestCase):
    """Simple test for extreme values of persistent entropy."""

    def test(self):
        # The entropy of a diagram with equal-length bars should be the
        # binary logarithm of the number of its samples.
        pd_1 = PersistenceDiagram(pairs=[
            (0.0, 1.0),
            (1.0, 2.0),
            (2.0, 3.0),
            (7.0, 8.0)
        ])

        entropy_1 = persistent_entropy(pd_1)

        # This is technically not required, but let's assume that there
        # might be some issues with the precision.
        self.assertAlmostEqual(entropy_1, 2.0)

        pd_2 = PersistenceDiagram(pairs=[
            (0.0, 1.0),
            (1.0, 2.0),
            (2.0, 3.0),
            (0.1, 0.2),
            (0.2, 0.3),
            (0.3, 0.4)
        ])

        entropy_2 = persistent_entropy(pd_2)

        self.assertLess(entropy_2, np.log2(len(pd_2)))
        self.assertGreater(entropy_2, 0.0)
