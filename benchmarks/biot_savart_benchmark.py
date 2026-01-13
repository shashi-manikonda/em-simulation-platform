import time
import numpy as np
from sandalwood import MultivariateTaylorFunction
from sandalwood.backends.cosy.cosy_backend import CosyScope  # Import Scope
from em_app.solvers import serial_biot_savart, Backend  # Import Backend Enum
from em_app.sources import RingCoil

def run_biot_savart_benchmark(num_source_points, num_field_points, order=0):
    print(
        f"--- Benchmarking: {num_source_points} src x {num_field_points} fld (Order {order}) ---"
    )

    # 1. Setup the problem geometry (a single current ring)
    ring_radius = 1.0
    ring_center = np.array([0, 0, 0])
    ring_axis = np.array([0, 0, 1])

    coil = RingCoil(
        current=1.0,
        radius=ring_radius,
        num_segments=num_source_points,
        center_point=ring_center,
        axis_direction=ring_axis,
        use_mtf_for_segments=False,
    )

    # Extract/Prepare segments
    coil_centers, coil_lengths, coil_directions = coil.get_segments()
    
    # Ensure inputs are real (float64) to trigger Fast Path in solver
    # RingCoil geometry generation uses complex numbers, but results are effectively real.
    segments_np = np.array([c.to_numpy_array() for c in coil_centers]).real.astype(np.float64)
    lengths_np = np.array(coil_lengths).real.astype(np.float64)
    directions_np = np.array([d.to_numpy_array() for d in coil_directions]).real.astype(np.float64)

    field_points = np.linspace([-1, -1, -1], [1, 1, 1], num_field_points)

    # Use Scope to manage memory
    with CosyScope():
        # 1. Benchmark COSY Backend
        duration_cosy = 0
        try:
            start = time.perf_counter()
            # Use Backend.COSY (auto-dispatches to Hybrid Fast Path for floats)
            _ = serial_biot_savart(
                segments_np,
                lengths_np,
                directions_np,
                field_points,
                order=order,
                backend=Backend.COSY,
            )
            duration_cosy = time.perf_counter() - start
            print(f"COSY Backend Time:   {duration_cosy:.6f} s")
        except Exception as e:
            print(f"COSY Backend Failed: {e}")

        # 2. Benchmark Python Backend
        start = time.perf_counter()
        _ = serial_biot_savart(
            segments_np,
            lengths_np,
            directions_np,
            field_points,
            order=order,
            backend=Backend.PYTHON,
        )
        duration_py = time.perf_counter() - start
        print(f"Python Backend Time: {duration_py:.6f} s")

        # Reporting
        if duration_cosy > 0:
            print(f"Speedup:             {duration_py / duration_cosy:.2f}x")
        print("-" * 40)


if __name__ == "__main__":
    # Initialize ONCE
    MultivariateTaylorFunction.initialize_mtf(max_order=10, max_dimension=4)

    # Run tests
    run_biot_savart_benchmark(100, 100)
    run_biot_savart_benchmark(1000, 1000)
    # run_biot_savart_benchmark(1000, 10000) # Optional larger case
