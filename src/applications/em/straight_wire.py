# mtflib/applications/em/straight_wire.py
"""
Function to calculate the magnetic field of a straight wire segment.
"""
import sys
import numpy as np
from .biot_savart import serial_biot_savart
from mtflib import MultivariateTaylorFunction

def straight_wire_field(start_point, end_point, current, field_points, num_segments=100):
    """
    Calculates the magnetic field from a straight, finite wire segment using
    numerical integration of the Biot-Savart law.

    The wire is defined by its start and end points in 3D space. The function
    approximates the wire as a series of small, discrete segments and sums their
    contributions.

    This function supports both numerical (NumPy array) and Taylor series (MTF object)
    evaluation points.

    Args:
        start_point (numpy.ndarray): (3,) array for the starting point of the wire.
        end_point (numpy.ndarray): (3,) array for the ending point of the wire.
        current (float): The current flowing through the wire.
        field_points (numpy.ndarray or array of MTF objects): (M, 3) array of points
            where the magnetic field is to be calculated.
        num_segments (int, optional): The number of segments to discretize the
            wire into for the numerical integration. Defaults to 100.

    Returns:
        numpy.ndarray or array of MTF objects: (M, 3) array representing the
            magnetic field vector (Bx, By, Bz) at each field point.
    """
    start_point = np.array(start_point)
    end_point = np.array(end_point)

    wire_vector = end_point - start_point
    wire_length = np.linalg.norm(wire_vector)

    if wire_length == 0:
        # If the wire has no length, the field is zero.
        if isinstance(field_points[0][0], np.ndarray):
             return np.zeros_like(field_points, dtype=float)
        else: # Assume MTF objects
             return np.array([MultivariateTaylorFunction.from_constant(0.0) for _ in range(3)], dtype=object)

    wire_direction = wire_vector / wire_length

    segment_length = wire_length / num_segments

    # Create the center points of each segment along the wire
    segment_centers = np.array([start_point + wire_direction * (i + 0.5) * segment_length for i in range(num_segments)])

    # All segments have the same length and direction
    element_lengths = np.full(num_segments, segment_length)
    element_directions = np.tile(wire_direction, (num_segments, 1))

    # The underlying serial_biot_savart function contains a 0.5 factor on the
    # dl_vectors to support a workflow for integrating over parameterized curves.
    # This direct numerical summation does not involve that subsequent integration,
    # so the result is off by a factor of 0.5. We compensate for that here by
    # multiplying the final result by 2.0.
    b_field_unit_current = serial_biot_savart(segment_centers, element_lengths, element_directions, field_points)

    # Scale by the current and correct for the factor of 0.5 in the underlying function.
    return b_field_unit_current * current * 2.0
