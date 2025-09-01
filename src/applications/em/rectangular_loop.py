# mtflib/applications/em/rectangular_loop.py
"""
Function to calculate the magnetic field of a rectangular current loop.
"""
import sys
import numpy as np
from .straight_wire import straight_wire_field
from mtflib import MultivariateTaylorFunction

def rectangular_loop_field(p1, p2, p4, current, field_points, num_segments_per_side=25):
    """
    Calculates the magnetic field from an arbitrarily oriented rectangular loop.

    The loop is defined by three corner points p1, p2, and p4, where the
    sides are the vectors (p2 - p1) and (p4 - p1). These two vectors must be
    orthogonal. The fourth corner (p3) is calculated automatically.

    The current direction is p1 -> p2 -> p3 -> p4 -> p1.

    Args:
        p1 (array-like): The first corner of the rectangle.
        p2 (array-like): The second corner, defining the first side from p1.
        p4 (array-like): The fourth corner, defining the second side from p1.
        current (float): The current flowing through the loop.
        field_points (numpy.ndarray or array of MTF objects): (M, 3) array of points
            where the magnetic field is to be calculated.
        num_segments_per_side (int, optional): The number of segments to discretize
            each side of the rectangle into. Defaults to 25.

    Returns:
        numpy.ndarray or array of MTF objects: (M, 3) array representing the
            magnetic field vector (Bx, By, Bz) at each field point.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p4 = np.array(p4)

    # Define the two side vectors from the corner p1
    side1 = p2 - p1
    side2 = p4 - p1

    # Check for orthogonality
    if not np.isclose(np.dot(side1, side2), 0):
        raise ValueError("The two vectors defining the sides from p1 must be orthogonal.")

    # Calculate the fourth corner p3
    p3 = p2 + side2

    corners = [p1, p2, p3, p4]

    total_b_field = None

    # Calculate and sum the B-field from each of the four sides
    for i in range(4):
        start_point = corners[i]
        end_point = corners[(i + 1) % 4]

        b_side = straight_wire_field(
            start_point, end_point, current, field_points, num_segments_per_side)

        if total_b_field is None:
            total_b_field = b_side
        else:
            total_b_field += b_side

    return total_b_field
