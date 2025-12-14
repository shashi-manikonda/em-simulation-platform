import numpy as np
import pytest
from mtflib import mtf

from em_app.vector_fields import FieldVector, VectorField

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


def test_fieldvector_initialization():
    """
    Test that FieldVector objects are initialized correctly for both numerical
    and MTF data.
    """
    # Test with numerical data
    x, y, z = 1.0, 2.0, 3.0
    fieldvector_numerical = FieldVector(x, y, z)
    assert isinstance(fieldvector_numerical.x, (float, int))
    assert np.allclose(fieldvector_numerical.to_numpy_array(), [1.0, 2.0, 3.0])

    # Test with MTF data
    try:
        x_mtf = mtf.var(1)
        y_mtf = mtf.var(2)
        z_mtf = mtf.var(3)
        fieldvector_mtf = FieldVector(x_mtf, y_mtf, z_mtf)
        assert fieldvector_mtf.is_mtf()
    except ImportError:
        pytest.skip("mtflib not installed, skipping MTF test.")


def test_vectorfield_magnitude():
    """
    Test that the VectorField class correctly calculates the magnitude of the field.
    """
    # Create a simple VectorField with numerical data
    field_points = np.array([[0, 0, 0], [1, 1, 1]])
    vectors_numerical = np.array([[1, 0, 0], [0, 1, 1]])

    # Wrap the vectors in FieldVector objects
    vectors_objects = np.array(
        [FieldVector(vec[0], vec[1], vec[2]) for vec in vectors_numerical], dtype=object
    )

    vector_field = VectorField(vectors_objects, field_points=field_points)

    # Calculate magnitude using the VectorField method
    magnitudes = vector_field.get_magnitude()

    # Expected magnitudes
    expected_magnitudes = np.array([1.0, np.sqrt(2)])

    assert np.allclose(magnitudes, expected_magnitudes)


def test_vectorfield_initialization_numerical():
    """
    Test VectorField initialization with numerical data.
    """
    field_points = np.array([[0, 0, 0], [1, 0, 0]])
    initial_vectors = np.array([[0, 0, 1], [0, 0, 2]])
    vector_field = VectorField(initial_vectors, field_points)

    points, vectors = vector_field._get_numerical_data()

    assert isinstance(vectors, np.ndarray)
    assert len(vectors) == 2
    assert np.allclose(vectors, initial_vectors)
    assert np.allclose(points, field_points)
