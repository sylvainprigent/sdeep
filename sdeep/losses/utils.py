"""Misc functions to calculate shapes"""

import numpy as np


def my_disk(center, radius):
    """Calculate the coordinates of points inside a disk

    Parameters
    ----------
    center: tuple
        (x, y) coordinates of the center
    radius: int
        Radius of the disk (in pixels)

    Return
    ------
    a tuple of the (xx, yy) points coordinates

    """
    s_r = radius * radius
    c_c = []
    r_r = []
    for i in range(center[0] - radius, center[0] + radius + 1):
        for j in range(center[1] - radius, center[1] + radius + 1):
            if (i - center[0]) * (i - center[0]) + (j - center[1]) * (
                    j - center[1]) <= s_r:
                r_r.append(i)
                c_c.append(j)
    return np.array(c_c), np.array(r_r)


def my_disk_ring(center, radius, alpha):
    """Calculate the coordinates of points inside a disk ring

    Parameters
    ----------
    center: tuple
        (x, y) coordinates of the center
    radius: int
        Radius of the disk (in pixels)
    alpha: int
        Width of the ring in pixels

    Return
    ------
    a tuple of the (xx, yy) points coordinates

    """

    s_r = radius * radius
    s_a = (radius + alpha) * (radius + alpha)
    r_o = radius + alpha + 1
    c_c = []
    r_r = []
    for i in range(center[0] - r_o, center[0] + r_o + 1):
        for j in range(center[1] - r_o, center[1] + r_o + 1):
            value = (i - center[0]) * (i - center[0]) + (j - center[1]) * (
                        j - center[1])
            if s_r < value <= s_a:
                r_r.append(i)
                c_c.append(j)
    return np.array(c_c), np.array(r_r)


def disk_patch(radius):
    """Create a patch with a disk shape where intensities sum to one

    Parameters
    ----------
    radius: int
        Radius of the disk (in pixels)

    """
    inner_patch = np.zeros((2 * radius + 1, 2 * radius + 1))
    irr, icc = my_disk((radius, radius), radius)
    inner_patch[irr, icc] = 1
    inner_patch /= np.sum(inner_patch)
    return inner_patch


def ring_patch(radius, alpha):
    """Create a patch with a ring shape where intensities sum to one

    Parameters
    ----------
    radius: int
        Radius of the disk (in pixels)
    alpha: int
        Width of the ring in pixels

    """
    outer_patch = np.zeros((2 * (radius + alpha) + 1, 2 * (radius + alpha) + 1))
    orr, occ = my_disk_ring((radius + alpha, radius + alpha), radius, alpha)
    outer_patch[orr, occ] = 1
    outer_patch /= np.sum(outer_patch)
    return outer_patch
