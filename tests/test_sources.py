import numpy as np
import pytest
from em_app.sources import RectangularCoil, RingCoil

# Global settings for tests
MAX_ORDER = 5
MAX_DIMENSION = 4
ETOL = 1e-20


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

from em_app.sources import _rotation_matrix_align_vectors

def test_rotation_matrix_align_vectors_identical():
    """Test aligning a vector with itself returns the identity matrix."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    rot_mat = _rotation_matrix_align_vectors(v1, v2)
    assert np.allclose(rot_mat, np.eye(3))
    assert np.allclose(rot_mat @ v1, v2)

def test_rotation_matrix_align_vectors_opposite():
    """Test aligning opposite vectors returns a 180 degree rotation."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([-1.0, 0.0, 0.0])
    rot_mat = _rotation_matrix_align_vectors(v1, v2)
    assert np.allclose(rot_mat @ v1, v2)

    # Different axis
    v1 = np.array([0.0, 1.0, 0.0])
    v2 = np.array([0.0, -1.0, 0.0])
    rot_mat = _rotation_matrix_align_vectors(v1, v2)
    assert np.allclose(rot_mat @ v1, v2)

    # Different axis 2
    v1 = np.array([0.0, 0.0, 1.0])
    v2 = np.array([0.0, 0.0, -1.0])
    rot_mat = _rotation_matrix_align_vectors(v1, v2)
    assert np.allclose(rot_mat @ v1, v2)

    # Arbitrary opposite
    v1 = np.array([1.0, 2.0, 3.0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = -v1
    rot_mat = _rotation_matrix_align_vectors(v1, v2)
    assert np.allclose(rot_mat @ v1, v2)

def test_rotation_matrix_align_vectors_orthogonal():
    """Test aligning orthogonal vectors."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    rot_mat = _rotation_matrix_align_vectors(v1, v2)
    assert np.allclose(rot_mat @ v1, v2)

    v1 = np.array([0.0, 0.0, 1.0])
    v2 = np.array([1.0, 0.0, 0.0])
    rot_mat = _rotation_matrix_align_vectors(v1, v2)
    assert np.allclose(rot_mat @ v1, v2)

def test_rotation_matrix_align_vectors_arbitrary():
    """Test aligning arbitrary non-orthogonal, non-opposite vectors."""
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([-1.0, 4.0, -2.0])
    rot_mat = _rotation_matrix_align_vectors(v1, v2)
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    assert np.allclose(rot_mat @ v1_norm, v2_norm)

def test_rotation_matrix_align_vectors_invalid_input():
    """Test that invalid inputs raise TypeError."""
    with pytest.raises(TypeError, match="v1 must be a 3-element NumPy array."):
        _rotation_matrix_align_vectors([1, 0, 0], np.array([0, 1, 0]))

    with pytest.raises(TypeError, match="v2 must be a 3-element NumPy array."):
        _rotation_matrix_align_vectors(np.array([1, 0, 0]), [0, 1, 0])

    with pytest.raises(TypeError, match="v1 must be a 3-element NumPy array."):
        _rotation_matrix_align_vectors(np.array([1, 0]), np.array([0, 1, 0]))

    with pytest.raises(TypeError, match="v2 must be a 3-element NumPy array."):
        _rotation_matrix_align_vectors(np.array([1, 0, 0]), np.array([0, 1]))

def test_rotation_matrix_align_vectors_zero_length():
    """Test zero length vectors."""
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="Vectors must not be zero length."):
        _rotation_matrix_align_vectors(v1, v2)

    with pytest.raises(ValueError, match="Vectors must not be zero length."):
        _rotation_matrix_align_vectors(v2, v1)
