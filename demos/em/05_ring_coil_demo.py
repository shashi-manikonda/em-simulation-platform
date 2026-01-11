import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from em_app.solvers import Backend, calculate_b_field
from em_app.sources import RingCoil
from sandalwood import mtf

# Ensure MTF is initialized with enough dimensions (4 needed for parameterization 'u')
mtf.initialize_mtf(max_order=4, max_dimension=4)


def analytic_b_field_on_axis(z, R, current, mu0=4 * np.pi * 1e-7):
    """
    Analytic formula for B-field on the axis of a current loop.
    Bz(z) = (mu0 * I * R^2) / (2 * (R^2 + z^2)^(3/2))
    """
    return (mu0 * current * R**2) / (2 * (R**2 + z**2) ** 1.5)


def run_verification():
    print("=== RingCoil Verification Demo ===")

    # Parameters
    R = 0.5  # Radius (m)
    current = 10.0  # Current (A)
    N_segments = 100
    z_points = np.linspace(-1, 1, 21)

    # Setup Coil
    coil = RingCoil(
        current=current,
        radius=R,
        num_segments=N_segments,
        center_point=np.array([0.0, 0.0, 0.0]),
        axis_direction=np.array([0.0, 0.0, 1.0]),
        use_mtf_for_segments=False,
    )

    # Field points along Z-axis
    field_points = np.zeros((len(z_points), 3))
    field_points[:, 2] = z_points

    # Analytic Solution
    B_analytic = analytic_b_field_on_axis(z_points, R, current)

    # Python Backend
    print("\nRunning Python Backend...")
    vf_py = calculate_b_field(coil, field_points, backend=Backend.PYTHON)
    B_py = vf_py.get_magnitude()

    # C++ Backend
    print("Running C++ Backend...")
    try:
        vf_cpp = calculate_b_field(coil, field_points, backend=Backend.CPP)
        B_cpp = vf_cpp.get_magnitude()
        cpp_success = True
    except Exception as e:
        print(f"C++ Backend Failed: {e}")
        cpp_success = False
        B_cpp = np.zeros_like(B_py)

    # Compare
    print("\nVerification Results:")
    print(
        f"{'z(m)':<10} {'Analytic(T)':<15} {'Python(T)':<15} {'C++(T)':<15} "
        f"{'Error(%)':<10}"
    )
    print("-" * 70)

    max_error_py = 0
    max_error_cpp = 0

    for i, z in enumerate(z_points):
        err_py = abs(B_py[i] - B_analytic[i]) / (B_analytic[i] + 1e-15) * 100000000
        # Scale for % but B is small so just relative
        err_py = abs(B_py[i] - B_analytic[i]) / (abs(B_analytic[i]) + 1e-15) * 100

        if cpp_success:
            err_cpp = abs(B_cpp[i] - B_analytic[i]) / (abs(B_analytic[i]) + 1e-15) * 100
        else:
            err_cpp = 100.0

        max_error_py = max(max_error_py, err_py)
        max_error_cpp = max(max_error_cpp, err_cpp)

        print(
            f"{z:<10.2f} {B_analytic[i]:<15.6e} {B_py[i]:<15.6e} {B_cpp[i]:<15.6e} "
            f"{err_py:<10.2f}"
        )

    print("-" * 70)
    print(f"Max Error (Python): {max_error_py:.4f}%")
    if cpp_success:
        print(f"Max Error (C++):    {max_error_cpp:.4f}%")
    else:
        print("Max Error (C++):    FAILED")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(z_points, B_analytic, "k-", label="Analytic")
    plt.plot(z_points, B_py, "bo", label="Python Backend", fillstyle="none")
    if cpp_success:
        plt.plot(z_points, B_cpp, "rx", label="C++ Backend")

    plt.xlabel("Z (m)")
    plt.ylabel("Bz (T)")
    plt.title(f"RingCoil On-Axis Field (R={R}m, I={current}A)")
    plt.legend()
    plt.grid(True)
    plt.savefig("ring_coil_verification.png")
    print("\nSaved plot to ring_coil_verification.png")


if __name__ == "__main__":
    run_verification()
