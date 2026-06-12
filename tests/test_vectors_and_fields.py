import numpy as np
import pytest
from em_app.vector_fields import FieldVector, VectorField
from sandalwood import mtf

# Global settings for tests
MAX_ORDER = 5
MAX_DIMENSION = 4
ETOL = 1e-20


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
        pytest.skip("sandalwood not installed, skipping MTF test.")


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

def test_fieldvector_cross_product():
    """
    Test the cross product of two FieldVector objects.
    """
    # Test standard basis vectors
    i = FieldVector(1.0, 0.0, 0.0)
    j = FieldVector(0.0, 1.0, 0.0)
    k = FieldVector(0.0, 0.0, 1.0)

    # i x j = k
    cross_ij = i.cross(j)
    assert np.allclose(cross_ij.to_numpy_array(), [0.0, 0.0, 1.0])
    assert isinstance(cross_ij, FieldVector)

    # j x k = i
    cross_jk = j.cross(k)
    assert np.allclose(cross_jk.to_numpy_array(), [1.0, 0.0, 0.0])

    # k x i = j
    cross_ki = k.cross(i)
    assert np.allclose(cross_ki.to_numpy_array(), [0.0, 1.0, 0.0])

    # Test arbitrary vectors
    v1 = FieldVector(2.0, 3.0, 4.0)
    v2 = FieldVector(5.0, 6.0, 7.0)

    # x = 3*7 - 4*6 = 21 - 24 = -3
    # y = 4*5 - 2*7 = 20 - 14 = 6
    # z = 2*6 - 3*5 = 12 - 15 = -3
    cross_v1v2 = v1.cross(v2)
    assert np.allclose(cross_v1v2.to_numpy_array(), [-3.0, 6.0, -3.0])

    # Cross product with itself should be zero vector
    cross_v1v1 = v1.cross(v1)
    assert np.allclose(cross_v1v1.to_numpy_array(), [0.0, 0.0, 0.0])

    # Anti-commutativity: v1 x v2 = -(v2 x v1)
    cross_v2v1 = v2.cross(v1)
    assert np.allclose(cross_v1v2.to_numpy_array(), -cross_v2v1.to_numpy_array())

    # Test TypeError
    with pytest.raises(TypeError, match="unsupported operand type\\(s\\) for cross product"):
        i.cross(1.0)
