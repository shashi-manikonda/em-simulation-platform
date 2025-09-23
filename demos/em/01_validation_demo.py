import numpy as np
import matplotlib.pyplot as plt
from em_app.sources import StraightWire, RectangularCoil
from em_app.solvers import calculate_b_field
from mtflib import mtf

mtf.initialize_mtf(max_order=6, max_dimension=4)

def analytical_b_field_straight_wire(current, a, z1, z2, z):
    """
    Analytical solution for the B-field of a finite straight wire.
    """
    mu_0 = 4 * np.pi * 1e-7
    cos_theta_1 = (z - z1) / np.sqrt((z - z1) ** 2 + a**2)
    cos_theta_2 = (z - z2) / np.sqrt((z - z2) ** 2 + a**2)
    b_phi_mag = (mu_0 * current / (4 * np.pi * a)) * (cos_theta_1 - cos_theta_2)
    return b_phi_mag

def main():
    # --- Part 1: Straight Wire Validation ---
    print("--- Straight Wire Validation ---")
    current = 1.0
    start_point = [0, 0, -1]
    end_point = [0, 0, 1]
    wire = StraightWire(current, start_point, end_point, num_segments=50)

    # Numerical calculation
    observation_point = np.array([[0.1, 0, 0]])
    b_field = calculate_b_field(wire, observation_point)
    b_field_numerical = b_field._b_vectors_mtf[0].to_numpy_array()
    print(f"Numerical B-field at {observation_point[0]}: {b_field_numerical}")

    # Analytical calculation
    b_field_analytical_mag = analytical_b_field_straight_wire(current, 0.1, -1, 1, 0)
    print(f"Analytical B-field magnitude: {b_field_analytical_mag}")

    # The B-field should be in the -y direction (phi direction)
    b_field_numerical_mag = np.linalg.norm(b_field_numerical)
    print(f"Numerical B-field magnitude: {b_field_numerical_mag}")

    error = abs(b_field_numerical_mag - b_field_analytical_mag) / b_field_analytical_mag
    print(f"Relative error: {error:.2%}")


    # --- Part 2: Geometry Visualization ---
    print("\n--- Geometry Visualization ---")
    # Plot straight wire
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    wire.plot(ax=ax1)
    ax1.set_title("Straight Wire Geometry")

    # Plot rectangular loop
    p1 = np.array([0, 0, 0])
    p2 = np.array([1, 0, 0])
    p4 = np.array([0, 1, 0])
    rect_loop = RectangularCoil(1.0, p1, p2, p4, 20)
    ax2 = fig.add_subplot(122, projection='3d')
    rect_loop.plot(ax=ax2)
    ax2.set_title("Rectangular Loop Geometry")

    plt.tight_layout()
    plt.savefig("01_geometries.png")
    plt.show()


if __name__ == "__main__":
    main()
