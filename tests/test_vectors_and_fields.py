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

def test_fieldvector_to_dataframe_numerical():
    """
    Test to_dataframe for FieldVector with numerical components.
    """
    import pandas as pd

    vec = FieldVector(1.5, 2.5, 3.5)
    df = vec.to_dataframe(["comp_x", "comp_y", "comp_z"])

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["comp_x", "comp_y", "comp_z"]
    assert len(df) == 1
    assert df["comp_x"][0] == 1.5
    assert df["comp_y"][0] == 2.5
    assert df["comp_z"][0] == 3.5


def test_fieldvector_to_dataframe_mtf():
    """
    Test to_dataframe for FieldVector with MTF components.
    """
    import pandas as pd

    try:
        # Already initialized by conftest.py
        x_mtf = mtf.var(1)
        y_mtf = mtf.var(2)
        z_mtf = mtf.var(3)

        vec = FieldVector(x_mtf, 2 * y_mtf, 3 * z_mtf)
        df = vec.to_dataframe(["x", "y", "z"])

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["x", "y", "z", "Order", "Exponents"]
        assert len(df) == 3

        dimension = mtf._MAX_DIMENSION

        # Check Exponents and corresponding coefficients
        exponents = [tuple(e) for e in df["Exponents"].tolist()]
        assert tuple([1, 0, 0] + [0] * (dimension - 3)) in exponents
        assert tuple([0, 1, 0] + [0] * (dimension - 3)) in exponents
        assert tuple([0, 0, 1] + [0] * (dimension - 3)) in exponents

        # Find index for each exponent
        idx_x = exponents.index(tuple([1, 0, 0] + [0] * (dimension - 3)))
        idx_y = exponents.index(tuple([0, 1, 0] + [0] * (dimension - 3)))
        idx_z = exponents.index(tuple([0, 0, 1] + [0] * (dimension - 3)))

        # Check coefficients
        assert df["x"][idx_x] == 1.0
        assert df["y"][idx_y] == 2.0
        assert df["z"][idx_z] == 3.0

        # Verify cross-terms are zero
        assert df["y"][idx_x] == 0.0
        assert df["z"][idx_x] == 0.0
        assert df["x"][idx_y] == 0.0
        assert df["z"][idx_y] == 0.0
        assert df["x"][idx_z] == 0.0
        assert df["y"][idx_z] == 0.0

    except ImportError:
        pytest.skip("sandalwood not installed, skipping MTF test.")
