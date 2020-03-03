"""Unit tests for Betti curves."""

import unittest

from pyper.representations import BettiCurve
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


class TestBettiCurveSimple(unittest.TestCase):
    def test(self):
        betti_curve = make_betti_curve(simple_diagram)
        print(betti_curve)
