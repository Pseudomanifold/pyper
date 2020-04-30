"""Persistence image class."""

import numpy as np


def _gaussian_kernel(x, y, ux, uy, sigma):
    """Gaussian kernel for a 2D point."""
    return 1 / (2 * np.pi * sigma**2) * np.exp(
        -((x - ux)**2 + (y - uy)**2) / (2 * sigma**2)
    )


def _transform(persistence_diagram):
    """Transform persistence diagram for persistence image calculation."""
    persistence_diagram[:, 1] -= persistence_diagram[:, 0]
    return persistence_diagram


class PersistenceImage:
    """Functor for creating a persistence image from a persistence diagram."""

    def __init__(self, resolution=(10, 10), **kwargs):
        """Create new persistence image with parameters."""
        # Set the width first as is custom for images. This is deviating
        # from the way we would handle images here.
        self.width, self.height = resolution

        # Access minimum and maximum coordinates for the functor, if
        # specified. They are used to ensure that *all* diagrams are
        # on the same scale.
        self.xmin = kwargs.get('xmin')
        self.xmax = kwargs.get('xmax')
        self.ymin = kwargs.get('ymin')
        self.ymax = kwargs.get('ymax')

    @staticmethod
    def w(y, b):
        """Weight function for an individual point in the persistence image.

        This is the default weight function described in the original
        paper [1]_.

        Parameters
        ----------
        y : float
            Persistence coordinate of the respective point.

        b : float
            Maximum persistence coordinate to use for the cut-off value
            of the weight function.

        Returns
        -------
        Weight for the specified point in the persistence image.

        [1]: H. Adams et al., 'Persistence Images: A Stable Vector
        Representation of Persistent Homology', Journal of Machine
        Learning Research 18 (2017), pp. 1-35.
        """
        if y <= 0:
            return 0.0
        elif 0 < y and y < b:
            return y / b
        else:
            return 1.0

    def transform(self, persistence_diagrams, max_persistence=None):
        """Calculate persistence images of a sequence of diagrams."""
        if max_persistence is None:
            max_persistence = np.max([
                abs(y - x) for pd in persistence_diagrams for x, y in pd
            ])

        return self._transform(
                    [pd for pd in persistence_diagrams],
                    max_persistence
                )

    def _transform(self, persistence_diagram, max_persistence):
        """Calculate persistence image of a single diagram.

        Parameters
        ----------
        persistence_diagram : `PersistenceDiagram` or `np.array`
            Persistence diagram whose persistence image should be
            calculated. This can also be a data type that one can
            convert to a 2D `numpy` array.

        max_persistence : float
            Specifies the maximum persistence value to consider for the
            conversion process. This is required for weighting points.

        Returns
        -------
        Persistence image.
        """
        # This converts *and* copies the input persistence diagram,
        # which is exactly what we need.
        persistence_diagram = _transform(np.array(persistence_diagram))

        xmin, ymin = self.xmin, self.ymin
        xmax, ymax = self.xmax, self.ymax

        if xmin is None:
            xmin = np.min(persistence_diagram[:, 0])

        if xmax is None:
            xmax = np.max(persistence_diagram[:, 0])

        if ymin is None:
            ymin = np.min(persistence_diagram[:, 1])

        if ymax is None:
            ymax = np.max(persistence_diagram[:, 1])

        # Create the proper grid for the persistence image. This will be
        # 'flipped' later on when it comes to the matrix conversion.
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

        # TODO: should we flip this here?
        persistence_image = np.zeros((self.height, self.width))

        for i in range(self._width):
            for j in range(self._height):
                cx = xc[i, j]
                cy = yc[i, j]

                value = 0.0

                # Evaluate the kernel function for *every* point of the
                # discretized domain.
                for c, d in persistence_diagram:
                    f = PersistenceImage.w(c, d, max_persistence) \
                      * _gaussian_kernel(cx, cy, c, d, 0.2)

                    value += f

                persistence_image[i, j] = value

        return persistence_image
