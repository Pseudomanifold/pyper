"""Unit tests for Betti curves."""

import unittest

from pyper.representations import make_betti_curve
from pyper.representations import PersistenceDiagram


simple_diagram = PersistenceDiagram(pairs=[
    (0.0, 1.0),
    (0.0, 6.0),
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 6.0),
    (5.0, 8.0)
])

class TestBettiCurveEdgeCases(unittest.TestCase):
    def test_empty(self):
        betti_curve = make_betti_curve([])

    def test_all_equal(self):
        betti_curve = make_betti_curve([[0.0, 0.0], [0.0, 0.0]])


class TestBettiCurveSimple(unittest.TestCase):
    def test(self):
        betti_curve = make_betti_curve(simple_diagram)

        # Change points of the Betti curve. If this does not work,
        # something is *really* wrong.
        self.assertEqual(betti_curve(0), 2)
        self.assertEqual(betti_curve(1), 2)
        self.assertEqual(betti_curve(2), 2)
        self.assertEqual(betti_curve(3), 2)
        self.assertEqual(betti_curve(4), 2)
        self.assertEqual(betti_curve(5), 3)
        self.assertEqual(betti_curve(6), 1)
        self.assertEqual(betti_curve(7), 1)
        self.assertEqual(betti_curve(8), 0)

        # Check something in-between two points; this should just be the
        # same value as above.
        self.assertEqual(betti_curve(1.50), 2)
        self.assertEqual(betti_curve(1.75), 2)
        self.assertEqual(betti_curve(5.50), 3)
        self.assertEqual(betti_curve(5.95), 3)
        self.assertEqual(betti_curve(7.25), 1)

        # Check whether the support is compact if we go outside the
        # specified domain of the function.
        self.assertEqual(betti_curve(-0.1), 0)
        self.assertEqual(betti_curve(9.1), 0)


