import numpy as np
import pytest
from em_app.vector_fields import FieldVector, VectorField
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
        from mtflib import mtf
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


def test_fieldvector_calculus():
    """
    Test the curl, divergence, and gradient methods of the FieldVector class.
    """
    try:
        from mtflib import mtf
        x, y, z = mtf.var(1), mtf.var(2), mtf.var(3)

        # Define a simple vector field B = [2x, 3y, 4z]
        b_field = FieldVector(2 * x, 3 * y, 4 * z)

        # Test divergence: div(B) = 2 + 3 + 4 = 9
        divergence = b_field.divergence()
        assert isinstance(divergence, mtf)
        assert np.isclose(divergence.get_constant(), 9.0)

        # Test curl: curl(B) should be zero for this field
        curl = b_field.curl()
        assert np.allclose(curl.to_numpy_array(), [0, 0, 0])

        # Test gradient: grad(B) should be a diagonal matrix
        gradient = b_field.gradient()
        expected_gradient = np.diag([2, 3, 4])

        # Convert the MTF gradient matrix to a numerical one
        numerical_gradient = np.zeros_like(expected_gradient, dtype=float)
        for i in range(3):
            for j in range(3):
                numerical_gradient[i, j] = gradient[i, j].get_constant()

        assert np.allclose(numerical_gradient, expected_gradient)

    except ImportError:
        pytest.skip("mtflib not installed, skipping MTF test.")