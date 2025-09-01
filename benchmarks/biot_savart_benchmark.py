import numpy as np
import time
import cProfile
import pstats
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mtflib.backends.cpp import mtf_cpp
from mtflib.taylor_function import MultivariateTaylorFunction
from src.applications.em.biot_savart import serial_biot_savart
from src.applications.em.current_ring import current_ring

def run_biot_savart_benchmark(num_source_points, num_field_points, order=0):
    """
    Benchmarks the performance of serial_biot_savart with both Python and C++ backends.
    """
    print(f"--- Benchmarking serial_biot_savart with {num_source_points} source segments and {num_field_points} field points ---")

    # 1. Setup the problem geometry (a single current ring)
    ring_radius = 1.0
    ring_center = np.array([0, 0, 0])
    ring_axis = np.array([0, 0, 1])

    segments_np, lengths_np, directions_np = current_ring(
        ring_radius, num_source_points, ring_center, ring_axis, return_mtf=False
    )

    field_points = np.linspace(
        [-1, -1, -1], [1, 1, 1], num_field_points
    )

    # 2. Benchmark the C++ implementation
    start_time_cpp = time.perf_counter()
    _ = serial_biot_savart(segments_np, lengths_np, directions_np, field_points, order=order, backend='cpp')
    end_time_cpp = time.perf_counter()
    duration_cpp = end_time_cpp - start_time_cpp
    print(f"C++ Backend Time: {duration_cpp:.6f} seconds")

    # 3. Benchmark the C++ V2 implementation
    start_time_cpp_v2 = time.perf_counter()
    _ = serial_biot_savart(segments_np, lengths_np, directions_np, field_points, order=order, backend='cpp_v2')
    end_time_cpp_v2 = time.perf_counter()
    duration_cpp_v2 = end_time_cpp_v2 - start_time_cpp_v2
    print(f"C++ V2 Backend Time: {duration_cpp_v2:.6f} seconds")

    # 4. Benchmark the C implementation
    start_time_c = time.perf_counter()
    _ = serial_biot_savart(segments_np, lengths_np, directions_np, field_points, order=order, backend='c')
    end_time_c = time.perf_counter()
    duration_c = end_time_c - start_time_c
    print(f"C Backend Time: {duration_c:.6f} seconds")

    # 5. Benchmark the Python implementation
    start_time_py = time.perf_counter()
    _ = serial_biot_savart(segments_np, lengths_np, directions_np, field_points, order=order, backend='python')
    end_time_py = time.perf_counter()
    duration_py = end_time_py - start_time_py
    print(f"Python Backend Time: {duration_py:.6f} seconds")

    # 6. Compare and report
    speedup_v1 = duration_py / duration_cpp if duration_cpp > 0 else float('inf')
    speedup_v2 = duration_py / duration_cpp_v2 if duration_cpp_v2 > 0 else float('inf')
    speedup_c = duration_py / duration_c if duration_c > 0 else float('inf')
    print(f"Speedup (Python/C++ V1): {speedup_v1:.2f}x")
    print(f"Speedup (Python/C++ V2): {speedup_v2:.2f}x")
    print(f"Speedup (Python/C): {speedup_c:.2f}x")
    print("----------------------------------\n")

if __name__ == '__main__':
    # Initialize MTF global settings once for consistency
    MultivariateTaylorFunction.initialize_mtf(max_order=10, max_dimension=3)

    # Run the benchmark for different problem sizes
    run_biot_savart_benchmark(num_source_points=100, num_field_points=100)
    run_biot_savart_benchmark(num_source_points=1000, num_field_points=100)
    run_biot_savart_benchmark(num_source_points=100, num_field_points=1000)
    run_biot_savart_benchmark(num_source_points=1000, num_field_points=1000)
