import numpy as np
import pytest
from em_app.currentcoils import RingCoil
from em_app.magneticfield import Bvec
from em_app.biot_savart import mu_0_4pi
from mtflib import mtf, ComplexMultivariateTaylorFunction

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
    b_vectors_objects = ring_coil.biot_savart(field_points)

    # Convert the list of Bvec objects to a single numerical NumPy array
    # This will preserve the complex parts if they exist
    b_vectors_numerical = np.array([b.to_numpy_array() for b in b_vectors_objects])

    # Analytical solution for the B-field on the axis of a circular loop
    # $B_{z} = \frac{\mu_0 I R^2}{2 (R^2 + z^2)^{3/2}}$
    mu_0 = mu_0_4pi * 4 * np.pi
    b_z_analytical = (
        mu_0 * current * radius**2
    ) / (2 * (radius**2 + z_points**2) ** 1.5)
    print("Numerical B-field (z-component):", np.real(b_vectors_numerical[:, 2]))
    print("Analytical B-field (z-component):", b_z_analytical)

    # Compare the real part of the z-component with the analytical solution
    assert np.allclose(np.real(b_vectors_numerical[:, 2]), b_z_analytical, rtol=1e-12)

    # Check that the imaginary part of the z-component is negligible
    assert np.allclose(np.imag(b_vectors_numerical[:, 2]), 0, atol=1e-12)

    # Check that both the real and imaginary parts of the x and y components are negligible
    assert np.allclose(b_vectors_numerical[:, 0], 0, atol=1e-12)
    assert np.allclose(b_vectors_numerical[:, 1], 0, atol=1e-12)


def test_biot_savart_with_mtf():
    """
    Test the Biot-Savart calculation when a current is a Multivariate Taylor Function.
    This is a basic check for functionality and to ensure the correct return type
    when complex coefficients are expected.
    """
    # Define coil parameters
    radius = 1.0
    num_segments = 40
    center = np.array([0, 0, 0])
    axis = np.array([0, 0, 1])
    field_point = np.array([[0, 0, 0.5]])

    # Initialize a MTF variable for the current
    current_mtf = 1.0+mtf.var(1)

    # Create a coil with an MTF current
    ring_coil = RingCoil(current_mtf, radius, num_segments, center, axis)

    # Calculate the B-field
    b_vectors_mtf = ring_coil.biot_savart(field_point)

    # Check if the result is an array of Bvec objects
    assert len(b_vectors_mtf) == 1
    assert isinstance(b_vectors_mtf[0], Bvec)

    # Check that the components of the returned Bvec are MTFs
    assert b_vectors_mtf[0].is_mtf()

    # The result should be a ComplexMultivariateTaylorFunction, as the current is a variable
    assert isinstance(b_vectors_mtf[0].Bx, mtf)
    assert isinstance(b_vectors_mtf[0].By, mtf)
    assert isinstance(b_vectors_mtf[0].Bz, mtf)

    # Check that the numerical values (zeroth-order coefficients) of the result
    # match the analytical solution for a current of 1.0
    mu_0 = mu_0_4pi * 4 * np.pi
    z = field_point[0, 2]
    b_z_analytical = (
        mu_0 * 1.0 * radius**2
    ) / (2 * (radius**2 + z**2) ** 1.5)

    assert np.isclose(b_vectors_mtf[0].Bz.extract_coefficient(tuple([0] * b_vectors_mtf[0].Bz.dimension)).item(), b_z_analytical)
    assert np.isclose(b_vectors_mtf[0].Bx.extract_coefficient(tuple([0] * b_vectors_mtf[0].Bx.dimension)).item(), 0)
    assert np.isclose(b_vectors_mtf[0].By.extract_coefficient(tuple([0] * b_vectors_mtf[0].By.dimension)).item(), 0)
