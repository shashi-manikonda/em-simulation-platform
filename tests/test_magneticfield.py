import numpy as np
import pytest
from em_app.magneticfield import Bvec, Bfield
from mtflib import mtf

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

    bfield = Bfield(b_vectors_objects, field_points=field_points)

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
    bfield = Bfield(b_vectors, field_points)

    points, vectors = bfield._get_numerical_data()

    assert isinstance(vectors, np.ndarray)
    assert len(vectors) == 2
    assert np.allclose(vectors, b_vectors)
    assert np.allclose(points, field_points)