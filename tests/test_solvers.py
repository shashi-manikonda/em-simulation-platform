import numpy as np
import pytest
from mtflib import mtf

from em_app.solvers import calculate_b_field, mu_0_4pi
from em_app.sources import RingCoil
from em_app.vector_fields import FieldVector

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


def test_biot_savart_ring_on_axis():
    """
    Test the magnetic field calculation on the axis of a circular current loop.

    This test now handles the possibility of complex coefficients from the
    Biot-Savart calculation. It asserts that the real part of the z-component
    matches the analytical solution and that both the real and imaginary parts
    of the transverse components are close to zero.
    """
    # Define a circular loop with known analytical solution
    radius = 1.0
    current = 1.0
    num_segments = 40  # High number for good approximation
    center = np.array([0, 0, 0])
    axis = np.array([0, 0, 1])

    ring_coil = RingCoil(current, radius, num_segments, center, axis)

    # Define a set of points along the z-axis (the coil's axis)
    z_points = np.linspace(-2, 2, 5)
    field_points = np.array([[0, 0, z] for z in z_points])

    # Calculate the magnetic field using the module
    b_field = calculate_b_field(ring_coil, field_points)
    b_vectors_objects = b_field._vectors_mtf

    # Convert the list of Bvec objects to a single numerical NumPy array
    # This will preserve the complex parts if they exist
    b_vectors_numerical = np.array([b.to_numpy_array() for b in b_vectors_objects])

    # Analytical solution for the B-field on the axis of a circular loop
    # $B_{z} = \frac{\mu_0 I R^2}{2 (R^2 + z^2)^{3/2}}$
    mu_0 = mu_0_4pi * 4 * np.pi
    b_z_analytical = (mu_0 * current * radius**2) / (
        2 * (radius**2 + z_points**2) ** 1.5
    )
    print("Numerical B-field (z-component):", np.real(b_vectors_numerical[:, 2]))
    print("Analytical B-field (z-component):", b_z_analytical)

    # Compare the real part of the z-component with the analytical solution
    assert np.allclose(np.real(b_vectors_numerical[:, 2]), b_z_analytical, rtol=1e-12)

    # Check that the imaginary part of the z-component is negligible
    assert np.allclose(np.imag(b_vectors_numerical[:, 2]), 0, atol=1e-12)

    # Check that both the real and imaginary parts of the x and y components
    # are negligible
    assert np.allclose(b_vectors_numerical[:, 0], 0, atol=1e-12)
    assert np.allclose(b_vectors_numerical[:, 1], 0, atol=1e-12)


def test_biot_savart_with_mtf():
    """
    Test the Biot-Savart calculation when a current is a MTF.

    This is a basic check for functionality and to ensure the correct return
    type when complex coefficients are expected.
    """
    # Define coil parameters
    radius = 1.0
    num_segments = 40
    center = np.array([0, 0, 0])
    axis = np.array([0, 0, 1])
    field_point = np.array([[0, 0, 0.5]])

    # Initialize a MTF variable for the current
    current_mtf = 1.0 + mtf.var(1)

    # Create a coil with an MTF current
    ring_coil = RingCoil(current_mtf, radius, num_segments, center, axis)

    # Calculate the B-field
    b_field = calculate_b_field(ring_coil, field_point)
    b_vectors_mtf = b_field._vectors_mtf

    # Check if the result is an array of FieldVector objects
    assert len(b_vectors_mtf) == 1
    assert isinstance(b_vectors_mtf[0], FieldVector)

    # Check that the components of the returned FieldVector are MTFs
    assert b_vectors_mtf[0].is_mtf()

    # The result should be a MTF, as the current is a variable
    assert isinstance(b_vectors_mtf[0].x, mtf)
    assert isinstance(b_vectors_mtf[0].y, mtf)
    assert isinstance(b_vectors_mtf[0].z, mtf)

    # Check that the numerical values (zeroth-order coefficients) of the result
    # match the analytical solution for a current of 1.0
    mu_0 = mu_0_4pi * 4 * np.pi
    z = field_point[0, 2]
    b_z_analytical = (mu_0 * 1.0 * radius**2) / (2 * (radius**2 + z**2) ** 1.5)

    z_component = b_vectors_mtf[0].z
    z_coeff = z_component.extract_coefficient(tuple([0] * z_component.dimension))
    assert np.isclose(z_coeff.item(), b_z_analytical)

    x_component = b_vectors_mtf[0].x
    x_coeff = x_component.extract_coefficient(tuple([0] * x_component.dimension))
    assert np.isclose(x_coeff.item(), 0)

    y_component = b_vectors_mtf[0].y
    y_coeff = y_component.extract_coefficient(tuple([0] * y_component.dimension))
    assert np.isclose(y_coeff.item(), 0)
