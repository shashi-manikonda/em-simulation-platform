import numpy as np
import pytest
from em_app.vector_fields import VectorFieldGrid, FieldVector

def test_vector_field_grid_init_and_reshape():
    """Verify VectorFieldGrid correctly reshapes points and vectors."""
    grid_shape = (2, 3)
    # 6 points total
    x, y = np.meshgrid(np.linspace(0, 1, 2), np.linspace(0, 1, 3), indexing='ij')
    field_points = np.stack([x.flatten(), y.flatten(), np.zeros(6)], axis=1)
    
    # Simple vectors
    vectors = np.random.rand(6, 3)
    
    vfg = VectorFieldGrid(vectors, field_points, grid_shape, ("x", "y"))
    
    # Check reshaped points
    reshaped_points = vfg.get_grid_points()
    assert reshaped_points.shape == (2, 3, 3)
    assert np.allclose(reshaped_points[0, 0], [0, 0, 0])
    assert np.allclose(reshaped_points[1, 2], [1, 1, 0])
    
    # Check reshaped vectors
    reshaped_vectors = vfg.get_grid_vectors()
    assert reshaped_vectors.shape == (2, 3, 3)
    assert np.allclose(reshaped_vectors.flatten(), vectors.flatten())

def test_vector_field_grid_validation():
    """Ensure VectorFieldGrid validates shape vs data size."""
    points = np.zeros((4, 3))
    vectors = np.zeros((4, 3))
    
    with pytest.raises(ValueError, match="number of field points must match the product"):
        VectorFieldGrid(vectors, points, (2, 3), ("x", "y"))
