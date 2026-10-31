"""RG2 linkage geometry, in meters/radians, shared by simulation and FK."""

import numpy as np


LINK_Y = .039592
LINK_Z = .038177
PIVOT_Y = .017178
PAD_Y = .01894995
MAX_WIDTH = .1005  # Measured opening of this installation; below linkage maximum.


def width_to_angle(width):
    radius = np.hypot(LINK_Y, LINK_Z)
    return np.arctan2(LINK_Z, LINK_Y) - np.arccos(
        np.clip((np.asarray(width) / 2 - PIVOT_Y + PAD_Y) / radius, -1, 1))


def angles_to_width(angles):
    angles = np.asarray(angles)
    return float(np.sum(PIVOT_Y + LINK_Y * np.cos(angles)
                        + LINK_Z * np.sin(angles) - PAD_Y))


def linkage_positions(width):
    angle = float(width_to_angle(width))
    return np.array([angle, -angle, angle, angle, -angle, angle])
