"""Persistence image class."""


import numpy as np


class PersistenceImageTransformer:
    '''
    Functor for creating a persistence image from a persistence diagram.
    '''

    def __init__(self, resolution, **kwargs):
        self._width = resolution    # x resolution
        self._height = resolution   # y resolution

        # Access minimum and maximum coordinates for the functor, if
        # specified. They are used to ensure that *all* diagrams are
        # on the same scale.
        self._xmin = kwargs.get('xmin')
        self._xmax = kwargs.get('xmax')
        self._ymin = kwargs.get('ymin')
        self._ymax = kwargs.get('ymax')

    @staticmethod
    def kernel(x, y, ux, uy, sigma):
        '''
        Kernel function for evaluating a single point. This cannot be
        changed for now.
        '''

        return 1 / (2 * math.pi * sigma**2) * np.exp(
            -((x - ux)**2 + (y - uy)**2) / (2 * sigma**2)
        )

    @staticmethod
    def w(x, y, b):
        '''
        Weight function for weighting an individual point in
        a persistence diagram.
        '''

        if y <= 0:
            return 0.0
        elif 0 < y and y < b:
            return y / b
        else:
            return 1.0

    def transform(self, persistence_diagram, max_persistence):

        points = np.array(persistence_diagram._pairs)

        xmin, ymin = self._xmin, self._ymin
        xmax, ymax = self._xmax, self._ymax

        if not xmin:
            xmin = np.min(points[:, 0])

        if not xmax:
            xmax = np.max(points[:, 0])

        if not ymin:
            ymin = np.min(points[:, 1])

        if not ymax:
            ymax = np.max(points[:, 1])

        x, xstep = np.linspace(xmin, xmax, self._width, retstep=True)
        y, ystep = np.linspace(ymin, ymax, self._height, retstep=True)

        xstep = np.repeat(xstep, self._width)
        ystep = np.repeat(ystep, self._height)

        xc, yc = np.meshgrid(
            np.cumsum(xstep) - 0.5 * xstep,
            np.cumsum(ystep) - 0.5 * ystep
        )

        # Add the respective minimum positions in order to shift the
        # origin of the coordinate system correctly.
        xc += xmin
        yc += ymin

        pit = np.zeros((self._width, self._height))

        for i in range(self._width):
            for j in range(self._height):
                cx = xc[i, j]
                cy = yc[i, j]

                value = 0.0

                # Evaluate the kernel function for *every* point of the
                # discretized domain.
                for c, d in points:
                    f = PersistenceImageTransform.w(c, d, max_persistence) \
                      * PersistenceImageTransform.kernel(cx, cy, c, d, 0.2)

                    value += f

                pit[i, j] = value

        return pit
