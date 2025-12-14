import numpy as np
import pytest
from mtflib import mtf

from em_app.sources import RectangularCoil, RingCoil

# Global settings for tests
MAX_ORDER = 5
MAX_DIMENSION = 4
ETOL = 1e-20


@pytest.fixture(scope="function", autouse=True)
def setup_function():
    mtf.initialize_mtf(max_order=MAX_ORDER, max_dimension=MAX_DIMENSION)
    mtf.set_etol(ETOL)
    global_dim = mtf.get_max_dimension()
    exponent_zero = tuple([0] * global_dim)
    yield global_dim, exponent_zero
    mtf._INITIALIZED = False


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
    centers_numerical = np.array([c.to_numpy_array() for c in coil.segment_centers])
    dist_from_center = np.linalg.norm(centers_numerical - center, axis=1)
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
