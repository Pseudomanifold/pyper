"""Statistics calculations for persistence diagrams.

This module contains additional summary statistics or auxiliary
functions that are useful when doing statistics with persistence
diagrams.
"""

import numpy as np


def _get_persistence_values(diagram):
    """Auxiliary function for calculating persistence values."""
    return [abs(x - y) for x, y in diagram]


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
