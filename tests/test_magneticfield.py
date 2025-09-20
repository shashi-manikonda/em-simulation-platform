import numpy as np
import pytest
from em_app import Bvec, Bfield

def test_bvec_initialization():
    """
    Test that Bvec objects are initialized correctly for both numerical
    and MTF data.
    """
    # Test with numerical data
    bx, by, bz = 1.0, 2.0, 3.0
    bvec_numerical = Bvec(bx, by, bz)
    assert isinstance(bvec_numerical.Bx, (float, int))
    assert np.allclose(bvec_numerical.to_numpy_array(), [1.0, 2.0, 3.0])

    # Test with MTF data
    try:
        from mtflib import mtf
        bx_mtf = mtf.var(1)
        by_mtf = mtf.var(2)
        bz_mtf = mtf.var(3)
        bvec_mtf = Bvec(bx_mtf, by_mtf, bz_mtf)
        assert bvec_mtf.is_mtf()
    except ImportError:
        pytest.skip("mtflib not installed, skipping MTF test.")


def test_bfield_magnitude():
    """
    Test that the Bfield class correctly calculates the magnitude of the field.
    """
    # Create a simple Bfield with numerical data
    field_points = np.array([[0, 0, 0], [1, 1, 1]])
    b_vectors_numerical = np.array([[1, 0, 0], [0, 1, 1]])
    
    # Wrap the vectors in Bvec objects
    b_vectors_objects = np.array(
        [Bvec(vec[0], vec[1], vec[2]) for vec in b_vectors_numerical], dtype=object
    )

    bfield = Bfield(field_points=field_points, b_vectors=b_vectors_objects)
    
    # Calculate magnitude using the Bfield method
    magnitudes = bfield.get_magnitude()
    
    # Expected magnitudes
    expected_magnitudes = np.array([1.0, np.sqrt(2)])

    assert np.allclose(magnitudes, expected_magnitudes)

def test_bfield_initialization_numerical():
    """
    Test Bfield initialization with numerical data.
    """
    field_points = np.array([[0, 0, 0], [1, 0, 0]])
    b_vectors = np.array([[0, 0, 1], [0, 0, 2]])
    bfield = Bfield(field_points, b_vectors)

    assert isinstance(bfield.b_vectors, np.ndarray)
    assert len(bfield.b_vectors) == 2
    assert isinstance(bfield.b_vectors[0], Bvec)