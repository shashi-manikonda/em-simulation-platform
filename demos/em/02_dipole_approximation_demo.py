import numpy as np
import matplotlib.pyplot as plt
from em_app.sources import RingCoil
from em_app.solvers import calculate_b_field
from mtflib import mtf

mtf.initialize_mtf(max_order=6, max_dimension=4)

def magnetic_dipole_b_field(magnetic_moment, r_vec):
    """
    Calculates the B-field from a magnetic dipole.
    """
    mu_0 = 4 * np.pi * 1e-7
    r_mag = np.linalg.norm(r_vec)
    r_hat = r_vec / r_mag

    term1 = 3 * np.dot(magnetic_moment, r_hat) * r_hat
    term2 = magnetic_moment

    b_field = (mu_0 / (4 * np.pi * r_mag**3)) * (term1 - term2)
    return b_field

def main():
    # --- Setup ---
    current = 1.0
    radius = 0.1
    ring_coil = RingCoil(current, radius, num_segments=20, center_point=np.array([0,0,0]), axis_direction=np.array([0,0,1]))

    # Magnetic moment of the ring coil
    area = np.pi * radius**2
    magnetic_moment = current * area * np.array([0, 0, 1])

    # --- Calculation ---
    distances = np.logspace(0, 3, 20) * radius # from 1 to 1000 radii
    errors = []

    for d in distances:
        observation_point = np.array([[0, 0, d]])

        # Full Biot-Savart calculation
        b_field = calculate_b_field(ring_coil, observation_point)
        b_field_numerical = b_field._b_vectors_mtf[0].to_numpy_array()

        # Dipole approximation
        b_field_dipole = magnetic_dipole_b_field(magnetic_moment, observation_point[0])

        # Calculate relative error
        error = np.linalg.norm(b_field_numerical - b_field_dipole) / np.linalg.norm(b_field_numerical)
        errors.append(error)

    # --- Visualization ---
    plt.figure(figsize=(8, 6))
    plt.loglog(distances / radius, errors, 'b-o')
    plt.title("Dipole Approximation Error vs. Distance")
    plt.xlabel("Distance from coil (in radii)")
    plt.ylabel("Relative Error")
    plt.grid(True, which="both", ls="--")
    plt.savefig("02_dipole_approximation_error.png")
    plt.show()


if __name__ == "__main__":
    main()
