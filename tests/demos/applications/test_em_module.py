# Tests for the EMLibrary module
import pytest
import numpy as np
import mtflib
from src.applications.em import biot_savart
from src.applications.em import current_ring
from mtflib import MultivariateTaylorFunction

@pytest.fixture(scope="function", autouse=True)
def setup_function():
    """Initialize MTF globals for each test in this module."""
    MultivariateTaylorFunction.initialize_mtf(max_order=5, max_dimension=4)
    MultivariateTaylorFunction.set_etol(1e-12)
    yield
    mtflib.taylor_function.MultivariateTaylorFunction._INITIALIZED = False


def test_numpy_biot_savart_single_loop():
    """
    Test the Biot-Savart calculation for a single circular loop.
    Compares the calculated B-field at the center with the analytical formula.
    B = (mu_0 * I) / (2 * R), assuming I=1A
    """
    ring_radius = 0.1  # meters
    num_segments = 100
    ring_center = np.array([0, 0, 0])
    ring_axis = np.array([0, 0, 1])
    field_points = np.array([[0, 0, 0]])

    # The current_ring function returns MTFs, but for this test, we need the evaluated values.
    # The biot_savart function expects numpy arrays.
    # The current_ring function is not suitable for generating the inputs for biot_savart directly.
    # I will generate the points manually.

    phi = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    d_phi = 2 * np.pi / num_segments
    element_lengths = ring_radius * d_phi * np.ones(num_segments)

    element_centers = np.zeros((num_segments, 3))
    element_centers[:, 0] = ring_radius * np.cos(phi)
    element_centers[:, 1] = ring_radius * np.sin(phi)

    element_directions = np.zeros((num_segments, 3))
    element_directions[:, 0] = -np.sin(phi)
    element_directions[:, 1] = np.cos(phi)


    # Calculate B-field using numpy_biot_savart
    b_field_numerical = biot_savart.numpy_biot_savart(
        element_centers,
        element_lengths,
        element_directions,
        field_points
    )

    # Analytical solution for B-field at the center of a loop
    # B = mu_0 * I / (2 * R), with I=1
    # mu_0 = 4 * pi * 1e-7
    # B = (4 * np.pi * 1e-7) / (2 * ring_radius) in the z-direction
    b_field_analytical_z = (4 * np.pi * 1e-7) / (2 * ring_radius)

    # The numpy_biot_savart function has a 0.5 factor for integration workflows,
    # so we must multiply by 2 for direct summation.
    expected_numerical_result = b_field_analytical_z

    b_field_numerical_x = b_field_numerical[0, 0]
    if isinstance(b_field_numerical_x, MultivariateTaylorFunction):
        b_field_numerical_x = b_field_numerical_x.extract_coefficient(tuple([0]*b_field_numerical_x.dimension)).item()

    b_field_numerical_y = b_field_numerical[0, 1]
    if isinstance(b_field_numerical_y, MultivariateTaylorFunction):
        b_field_numerical_y = b_field_numerical_y.extract_coefficient(tuple([0]*b_field_numerical_y.dimension)).item()

    b_field_numerical_z = b_field_numerical[0, 2]
    if isinstance(b_field_numerical_z, MultivariateTaylorFunction):
        b_field_numerical_z = b_field_numerical_z.extract_coefficient(tuple([0]*b_field_numerical_z.dimension)).item()

    assert np.allclose(b_field_numerical_x, 0)
    assert np.allclose(b_field_numerical_y, 0)
    assert np.allclose(b_field_numerical_z * 2, expected_numerical_result, rtol=1e-3)


def test_current_ring_output_shapes():
    """
    Test the output shapes and types of the current_ring function.
    """
    ring_radius = 0.1
    num_segments = 10
    ring_center = np.array([0, 0, 0])
    ring_axis = np.array([0, 0, 1])

    segment_mtfs, element_lengths, direction_vectors = current_ring.current_ring(
        ring_radius, num_segments, ring_center, ring_axis
    )

    assert isinstance(segment_mtfs, np.ndarray)
    assert segment_mtfs.shape == (num_segments, 3)
    assert isinstance(element_lengths, np.ndarray)
    assert element_lengths.shape == (num_segments,)
    assert isinstance(direction_vectors, np.ndarray)
    assert direction_vectors.shape == (num_segments, 3)

def test_current_ring_segment_positions():
    """
    Test that the generated segment centers lie on a circle of the correct radius.
    """
    ring_radius = 0.1
    num_segments = 10
    ring_center = np.array([0, 0, 0])
    ring_axis = np.array([0, 0, 1])

    segment_mtfs, _, _ = current_ring.current_ring(
        ring_radius, num_segments, ring_center, ring_axis
    )

    # We need to evaluate the MTFs to get the positions.
    # The `current_ring` function returns MTFs of the variables.
    # We can evaluate them at the origin (0,0,0,0) to get the center of the segments.

    for mtf in segment_mtfs:
        # The mtf is a vector of 3 MTF objects.
        # We need to evaluate each component.
        pos = np.array([c.eval([0,0,0,0])[0] for c in mtf])
        distance_from_center = np.linalg.norm(pos - ring_center)
        assert np.isclose(distance_from_center, ring_radius)

def test_current_ring_direction_orthogonality():
    """
    Test that the direction vector of each segment is orthogonal to its position vector.
    """
    ring_radius = 0.1
    num_segments = 10
    ring_center = np.array([0, 0, 0])
    ring_axis = np.array([0, 0, 1])

    segment_mtfs, _, direction_vectors_mtf = current_ring.current_ring(
        ring_radius, num_segments, ring_center, ring_axis
    )

    for i in range(num_segments):
        pos_mtf = segment_mtfs[i]
        dir_mtf = direction_vectors_mtf[i]

        # Evaluate at the origin to get the vectors
        pos_vec = np.array([c.eval([0,0,0,0])[0] for c in pos_mtf])
        dir_vec = np.array([c.eval([0,0,0,0])[0] for c in dir_mtf])

        # The position vector from the center of the ring
        pos_vec_from_center = pos_vec - ring_center

        # The dot product should be close to zero
        dot_product = np.dot(pos_vec_from_center, dir_vec)
        assert np.isclose(dot_product, 0)
