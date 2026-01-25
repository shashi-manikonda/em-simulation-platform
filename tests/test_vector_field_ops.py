import numpy as np
import pytest
from em_app.vector_fields import FieldVector, VectorField
from sandalwood import mtf

def test_vector_field_arithmetic():
    """Test addition, subtraction and scalar multiplication of VectorFields."""
    points = np.array([[0, 0, 0], [1, 1, 1]])
    vec1 = np.array([[1, 0, 0], [0, 1, 0]])
    vec2 = np.array([[0, 1, 0], [1, 0, 0]])
    
    vf1 = VectorField(vec1, points)
    vf2 = VectorField(vec2, points)
    
    # Addiction
    vf_sum = vf1 + vf2
    expected_sum = np.array([[1, 1, 0], [1, 1, 0]])
    _, num_sum = vf_sum._get_numerical_data()
    assert np.allclose(num_sum, expected_sum)
    
    # Subtraction
    vf_diff = vf1 - vf2
    expected_diff = np.array([[1, -1, 0], [-1, 1, 0]])
    _, num_diff = vf_diff._get_numerical_data()
    assert np.allclose(num_diff, expected_diff)
    
    # Scalar multiplication
    vf_scaled = vf1 * 2.5
    expected_scaled = vec1 * 2.5
    _, num_scaled = vf_scaled._get_numerical_data()
    assert np.allclose(num_scaled, expected_scaled)

def test_vector_field_soa_optimization():
    """Verify that SOA storage correctly handles components and reconstruction."""
    points = np.array([[0,0,0], [1,1,1]])
    vx = np.array([1.0, 4.0])
    vy = np.array([2.0, 5.0])
    vz = np.array([3.0, 6.0])
    
    # Initialize using components (triggers SOA)
    vf = VectorField((vx, vy, vz), points)
    assert vf._storage_mode == "soa"
    
    # Verify reconstruction
    expected = np.array([[1, 2, 3], [4, 5, 6]])
    _, num_data = vf._get_numerical_data()
    assert np.allclose(num_data, expected)

def test_vector_field_mtf_conversion():
    """Test handling of MTF vectors in VectorField."""
    points = np.array([[0,0,0]])
    # Create an MTF vector
    v_mtf = FieldVector(mtf.from_constant(1.0), mtf.from_constant(0.0), mtf.from_constant(0.0))
    vf = VectorField(np.array([v_mtf], dtype=object), points)
    
    # Check if storage mode is MTF
    assert vf._storage_mode == "aos_mtf"
    
    # Verify numerical extraction
    _, num_data = vf._get_numerical_data()
    assert np.allclose(num_data, [[1, 0, 0]])
