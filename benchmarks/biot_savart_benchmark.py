import time

import numpy as np
from sandalwood import MultivariateTaylorFunction

from em_app.solvers import serial_biot_savart
# from em_app.sources import current_ring


def run_biot_savart_benchmark(num_source_points, num_field_points, order=0):
    """
    Benchmarks the performance of serial_biot_savart with both Python and C++ backends.
    """
    print(
        f"--- Benchmarking serial_biot_savart with {num_source_points} source segments and {num_field_points} field points ---"
    )

    # 1. Setup the problem geometry (a single current ring)
    ring_radius = 1.0
    ring_center = np.array([0, 0, 0])
    ring_axis = np.array([0, 0, 1])

    # Create a RingCoil instance
    from em_app.sources import RingCoil
    coil = RingCoil(
        current=1.0, 
        radius=ring_radius, 
        num_segments=num_source_points, 
        center_point=ring_center, 
        axis_direction=ring_axis, 
        use_mtf_for_segments=False
    )
    
    # Extract segments (centers, lengths, directions)
    # The get_segments method returns MTF objects or numbers depending on use_mtf_for_segments
    # Since we set use_mtf_for_segments=False, these should be compatible with the benchmark
    # However, RingCoil stores them as Vector objects (likely), need to convert to numpy arrays if serial_biot_savart expects that.
    # Looking at solvers.py: serial_biot_savart expects numpy arrays of shape (N, 3), (N,), (N, 3).
    # Coil.get_segments returns (segment_centers, segment_lengths, segment_directions)
    # where centers/directions might be lists of Vectors or arrays.
    
    # Let's inspect how calculate_b_field handles it in solvers.py lines 61-66:
    # It converts to numpy arrays:
    # element_centers_np = np.array([c.to_numpy_array() for c in coil.segment_centers])
    # element_directions_np = np.array([d.to_numpy_array() for d in coil.segment_directions])
    
    # We should replicate that here to feed into serial_biot_savart.
    
    coil_centers, coil_lengths, coil_directions = coil.get_segments()
    
    segments_np = np.array([c.to_numpy_array() for c in coil_centers])
    # lengths are already numpy array of scalars usually? Let's assume so or convert.
    lengths_np = np.array(coil_lengths) 
    directions_np = np.array([d.to_numpy_array() for d in coil_directions])

    field_points = np.linspace([-1, -1, -1], [1, 1, 1], num_field_points)

    # 2. Benchmark the C++ implementation
    duration_cpp = 0
    try:
        start_time_cpp = time.perf_counter()
        _ = serial_biot_savart(
            segments_np, lengths_np, directions_np, field_points, order=order, backend="cpp"
        )
        end_time_cpp = time.perf_counter()
        duration_cpp = end_time_cpp - start_time_cpp
        print(f"C++ Backend Time: {duration_cpp:.6f} seconds")
    except Exception as e:
        print(f"C++ Backend Failed: {e}")

    # 3. Benchmark the C++ V2 implementation
    duration_cpp_v2 = 0
    try:
        start_time_cpp_v2 = time.perf_counter()
        _ = serial_biot_savart(
            segments_np,
            lengths_np,
            directions_np,
            field_points,
            order=order,
            backend="cpp_v2",
        )
        end_time_cpp_v2 = time.perf_counter()
        duration_cpp_v2 = end_time_cpp_v2 - start_time_cpp_v2
        print(f"C++ V2 Backend Time: {duration_cpp_v2:.6f} seconds")
    except Exception as e:
        print(f"C++ V2 Backend Failed: {e}")

    # 4. Benchmark the C implementation
    duration_c = 0
    try:
        start_time_c = time.perf_counter()
        _ = serial_biot_savart(
            segments_np, lengths_np, directions_np, field_points, order=order, backend="c"
        )
        end_time_c = time.perf_counter()
        duration_c = end_time_c - start_time_c
        print(f"C Backend Time: {duration_c:.6f} seconds")
    except Exception as e:
        print(f"C Backend Failed: {e}")

    # 5. Benchmark the Python implementation
    start_time_py = time.perf_counter()
    _ = serial_biot_savart(
        segments_np,
        lengths_np,
        directions_np,
        field_points,
        order=order,
        backend="python",
    )
    end_time_py = time.perf_counter()
    duration_py = end_time_py - start_time_py
    print(f"Python Backend Time: {duration_py:.6f} seconds")

    # 6. Compare and report
    speedup_v1 = duration_py / duration_cpp if duration_cpp > 0 else float("inf")
    speedup_v2 = duration_py / duration_cpp_v2 if duration_cpp_v2 > 0 else float("inf")
    speedup_c = duration_py / duration_c if duration_c > 0 else float("inf")
    print(f"Speedup (Python/C++ V1): {speedup_v1:.2f}x")
    print(f"Speedup (Python/C++ V2): {speedup_v2:.2f}x")
    print(f"Speedup (Python/C): {speedup_c:.2f}x")
    print("----------------------------------\n")


if __name__ == "__main__":
    # Initialize MTF global settings once for consistency
    MultivariateTaylorFunction.initialize_mtf(max_order=10, max_dimension=4)

    # Run the benchmark for different problem sizes
    run_biot_savart_benchmark(num_source_points=100, num_field_points=100)
    run_biot_savart_benchmark(num_source_points=1000, num_field_points=100)
    run_biot_savart_benchmark(num_source_points=100, num_field_points=1000)
    run_biot_savart_benchmark(num_source_points=1000, num_field_points=1000)
    
    print("\n--- High Order Benchmark (Order=3) ---")
    run_biot_savart_benchmark(num_source_points=100, num_field_points=100, order=3)
