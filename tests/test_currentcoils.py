import numpy as np
import pytest
from em_app.currentcoils import RingCoil, RectangularCoil
from em_app.magneticfield import Bvec
from em_app.biot_savart import mu_0_4pi
from mtflib import mtf, ComplexMultivariateTaylorFunction

def test_ring_coil_segment_generation():
    """
    Test if RingCoil correctly generates the specified number of segments
    with the correct properties.
    """
    current = 1.0
    radius = 2.0
    num_segments = 10
    center = np.array([1, 1, 1])
    axis = np.array([0, 0, 1])

    coil = RingCoil(current, radius, num_segments, center, axis)

    # Check the number of segments
    assert len(coil.segment_centers) == num_segments

    # Check the segment lengths
    expected_length = 2 * np.pi * radius / num_segments
    assert np.allclose(coil.segment_lengths, expected_length)

    # Check if the segment centers are at the correct distance from the center
    dist_from_center = np.linalg.norm(coil.segment_centers - center, axis=1)
    assert np.allclose(dist_from_center, radius)


def test_rectangular_coil_segment_generation():
    """
    Test if RectangularCoil generates the correct number of segments
    for each side.
    """
    current = 1.0
    p1 = np.array([0, 0, 0])
    p2 = np.array([1, 0, 0])
    p4 = np.array([0, 2, 0])
    num_segments_per_side = 5

    coil = RectangularCoil(current, p1, p2, p4, num_segments_per_side)

    # A rectangular coil has 4 sides
    expected_total_segments = 4 * num_segments_per_side
    assert len(coil.segment_centers) == expected_total_segments


def test_invalid_rectangular_coil_creation():
    """
    Test that RectangularCoil raises an error for non-orthogonal sides.
    """
    current = 1.0
    p1 = np.array([0, 0, 0])
    p2 = np.array([1, 0, 0])
    p4 = np.array([1, 1, 0])  # Not orthogonal to p1-p2 vector

    with pytest.raises(ValueError, match="Side vectors from p1 must be orthogonal."):
        RectangularCoil(current, p1, p2, p4, num_segments_per_side=5)