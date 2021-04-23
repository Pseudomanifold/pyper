"""Betti curves and functions for creating them."""

import math
import numbers

import numpy as np
import pandas as pd


def make_betti_curve(diagram):
    """Create a Betti curve from a persistence diagram.

    Creates a Betti curve of a persistence diagram, i.e. a curve that
    depicts the number of active intervals according to the threshold
    of the filtration.

    Parameters
    ----------
    diagram:
        Persistence diagram to convert

    Returns
    -------
    Betti curve of the input diagram, in the form of a `BettiCurve`
    instance.
    """
    # Contains all potential event points, i.e. points at which the
    # Betti curve might change.
    event_points = []

    for x, y in diagram:
        event_points.append((x, True))
        event_points.append((y, False))

    event_points = sorted(event_points, key=lambda x: x[0])
    n_active = 0

    output = []

    # Create the 'raw' sequence of event points first. This blindly
    # assumes that all creation and destruction times are different
    # from each other. If this is *not* the case, the same value is
    # used with a different number of active intervals. This may be
    # a problem for the consistency of indices later on.
    for p, is_generator in event_points:
        if is_generator:
            n_active += 1
        else:
            n_active -= 1

        output.append((p, n_active))

    # If the diagram is empty, skip everything. In the following, I will
    # assume that at least a single point exists.
    if not event_points:
        return None

    prev_p = event_points[0][0]   # Previous time point
    prev_v = 0                    # Previous number of active intervals

    # Will contain the tuples that give rise to the Betti curve in the
    # end, i.e. the threshold and the number of active intervals.
    output_ = []

    # Functor that is called to simplify the loop processing, which
    # requires one extra pass to handle the last interval properly.
    def process_event_points(p, n_active):

        # Admittedly, not the most elegant solution, but at least I do
        # not have to duplicate the loop body.
        nonlocal prev_p
        nonlocal prev_v
        nonlocal output_

        # Update the number of active intervals for as long as the
        # current threshold does *not* change.
        if prev_p == p:
            prev_v = n_active

        # Time point changed; the monotonically increasing subsequence
        # should now be stored.
        else:

            # Check whether this is *not* the first output and create
            # a transition point in the data set.
            if output_:

                # This makes the previous interval half-open by
                # introducing a fake transition point *between*
                # the existing points.
                old_value = output_[-1][1]
                old_point = np.nextafter(prev_p, prev_p - 1)

                # Inserts a fake point to obtain half-open intervals for
                # the whole function.
                output_.append((old_point, old_value))

            output_.append((prev_p, prev_v))

            prev_p = p
            prev_v = n_active

    for p, n_active in output:
        process_event_points(p, n_active)

    # Store the last subsequence if applicable. To this end, we need to
    # check if the last proper output was different from our previously
    # seen value. If so, there's another sequence in the output that we
    # missed so far.
    if len(output_) > 0 and prev_p != output_[-1][0]:

        # Note that the two arguments are fake; they are only required
        # to trigger the insertion of another interval.
        process_event_points(prev_p + 1, prev_v + 1)

    output = output_
    return BettiCurve(output)


class BettiCurve:
    """A Betti curve of a certain dimension.

    This class is the main representation of a Betti curve, i.e. a curve
    that contains the number of active topological features at every
    point of a filtration process.

    This class provides some required wrapper functions to simplify,
    and improve, the usage of this concept.
    """

    def __init__(self, values):
        """Create a new Betti curve from a sequence of values.

        Creates a new Betti curve from a sequence of values. The values
        are supposed to be ordered according to their filtration value,
        such that the first dimension represents the filtration axis.

        Parameters
        ----------
        values:
            Input values. This must be a sequence of tuples, with the
            first dimension representing the threshold of a function,
            and the second dimension representing the curve value. In
            the function itself, `pd.DataFrame` will be used.
        """
        if isinstance(values, pd.Series):
            self._data = values

            # It's brute force, but this ensures that the data frames
            # are compatible with each other.
            assert self._data.index.name == 'threshold'

        else:
            self._data = pd.DataFrame.from_records(
                values,
                columns=['threshold', 'n_features'],
                index='threshold'
            )['n_features']

    def __call__(self, threshold):
        """Evaluate the Betti curve at a given threshold.

        Parameters
        ----------
        threshold:
            Threshold at which to evaluate the curve. All numbers are
            valid here, but for some of them, the function may return
            zero.

        Returns
        -------
        Number of active features in the Betti curve under the given
        threshold.
        """
        match = self._data[self._data.index == threshold]
        if not match.empty:
            return match.values[0]

        # Interpolate between the nearest two indices. For most Betti
        # curves, this should be the same value anyway, but if one is
        # calculating averages, this might change.
        else:
            lower = self._data[self._data.index < threshold].index
            upper = self._data[self._data.index > threshold].index

            if not lower.empty and not upper.empty:
                # Take the *last* index of the lower half of the data,
                # and the *first* index of the upper half of the data,
                # in order to find the proper neighbours.
                lower = lower[-1]
                upper = upper[0]

                return 0.5 * (self._data[lower] + self._data[upper])
            else:
                # Either one of the indices is *outside* the halves of
                # the data, so we return zero because the curve has to
                # have compact support.
                return 0.0

    def __repr__(self):
        """Return a string-based representation of the curve."""
        return self._data.__repr__()

    def __add__(self, other):
        """Add a Betti curve to another Betti curve.

        Performs addition of two Betti curves. This necessitates
        re-indexing values accordingly in order to evaluate them
        properly.

        In case `other` is a number, does elementwise addition.

        Parameters
        ----------
        other:
            Betti curve to add to the current one, *or* a number, which
            is added to *all* values of the current Betti curve.

        Returns
        -------
        Betti curve that results from the addition.
        """
        if isinstance(other, numbers.Number):
            return BettiCurve(self._data + other)

        # Not a number, so let's re-index the Betti curve and perform
        # addition for the new curves.

        new_index = self._data.index.union(other._data.index)

        # The `fillna` is required because we might have a filtration
        # value that *precedes* the first index of one of the frames.
        left = self._data.reindex(new_index, method='ffill').fillna(0)
        right = other._data.reindex(new_index, method='ffill').fillna(0)

        return BettiCurve(left + right)

    def __radd__(self, other):
        """Arithmetic with Betti curves on the right-hand side."""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __neg__(self):
        """Negate the current Betti curve.

        Negates the current values of the Betti curves, i.e. applies
        a unary minus operation to the curve.

        Returns
        -------
        Negated Betti curve
        """
        return BettiCurve(-self._data)

    def __sub__(self, other):
        """Subtract another Betti curve from the current one."""
        return self.__add__(-other)

    def __abs__(self):
        """Calculate absolute value of the Betti curve.

        Calculates the absolute value of the Betti curve. Does not
        modify the current Betti curve.

        Returns
        -------
        Absolute value of the Betti curve
        """
        return BettiCurve(abs(self._data))

    def __truediv__(self, x):
        """Perform elementwise division of a Betti curve by some number.

        Parameters
        ----------
        x:
            Number to divide the Betti curve by

        Returns
        -------
        Betti curve divided by `x`
        """
        return BettiCurve(self._data / x)

    def norm(self, p=1.0):
        """$L_p$ norm calculation for the Betti curve.

        Calculates an $L_p$ norm of the Betti curve and returns the
        result.

        Parameters
        ----------
        p:
            Exponent for the corresponding $L_p$ norm

        Returns
        -------
        $L_p$ norm of the current Betti curve
        """
        result = 0.0
        for (x1, y1), (x2, y2) in zip(
                self._data.iteritems(),
                self._data.shift(axis='index').dropna().iteritems()):

            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1

            def evaluator(x):
                if m == 0.0:
                    return math.pow(c, p) * x
                else:
                    return math.pow(m*x + c, p+1) / (m * (p + 1))

            integral = abs(evaluator(x2) - evaluator(x1))
            result += integral

        return math.pow(result, 1.0 / p)

    def distance(self, other, p=1.0):
        """Calculate distance between two Betti curves.

        Calculates the distance between the current Betti curve and
        another one, subject to a certain $L_p$ norm.

        Parameters
        ----------
        other:
            Other Betti curve
        p:
            Exponent for the corresponding $L_p$ norm

        Returns
        -------
        Distance between the two curves. This is based on the $L_p$ norm
        of the difference curve.
        """
        return abs(self - other).norm(p)
