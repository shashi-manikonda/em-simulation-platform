
import time
import numpy as np
from sandalwood import MultivariateTaylorFunction
from sandalwood.backends.cosy.cosy_backend import CosyScope
from em_app.solvers import serial_biot_savart, Backend
from em_app.sources import RingCoil

def benchmark_scenario(n_field_points):
    """Run benchmark for a specific number of field points."""
    print(f"\nTarget: {n_field_points} field points")
    
    # Setup Source (Ring Coil)
    # Use reasonable resolution
    coil = RingCoil(
        current=1.0, 
        radius=1.0, 
        num_segments=1000, 
        center_point=[0,0,0], 
        axis_direction=[0,0,1],
        use_mtf_for_segments=False
    )
    c, l, d = coil.get_segments()
    
    # Convert to float arrays for fast path
    c_arr = np.array([x.to_numpy_array() for x in c]).real.astype(np.float64)
    l_arr = np.array(l).real.astype(np.float64)
    d_arr = np.array([x.to_numpy_array() for x in d]).real.astype(np.float64)
    
    # Setup Field Points (Random cloud in box)
    # Using random to avoid cache artifacts, but consistent seed
    np.random.seed(42)
    field_points = np.random.uniform(-2, 2, (n_field_points, 3)).astype(np.float64)
    
    # 1. Python Backend
    print("  Running Python Backend...", end="", flush=True)
    start_py = time.perf_counter()
    _ = serial_biot_savart(c_arr, l_arr, d_arr, field_points, backend=Backend.PYTHON)
    time_py = time.perf_counter() - start_py
    print(f" Done ({time_py:.4f} s)")
    
    # 2. COSY Backend
    print("  Running COSY Backend...", end="", flush=True)
    start_cosy = time.perf_counter()
    with CosyScope():
        _ = serial_biot_savart(c_arr, l_arr, d_arr, field_points, backend=Backend.COSY)
    time_cosy = time.perf_counter() - start_cosy
    print(f" Done ({time_cosy:.4f} s)")
    
    # 3. Stats
    speedup = time_py / time_cosy if time_cosy > 0 else 0
    print(f"  Speedup: {speedup:.2f}x")
    return n_field_points, time_py, time_cosy, speedup

def main():
    print("=== EM-Simulation-Platform Solver Benchmark ===")
    print("Comparing Python vs COSY Backend for Biot-Savart (SoA Optimized)")
    
    # Initialize MTF (Order 0 for float calculation, Order 1+ for derivatives if needed)
    # Using Order 0 for pure field calculation benchmark
    MultivariateTaylorFunction.initialize_mtf(max_order=1, max_dimension=3)
    
    results = []
    
    # Dry run to warm up JIT if any
    benchmark_scenario(100)
    
    # Scenarios
    scenarios = [10_000, 100_000, 1_000_000]
    
    for n in scenarios:
        results.append(benchmark_scenario(n))
        
    print("\n=== Summary ===")
    print(f"{'Points':<12} | {'Python (s)':<12} | {'COSY (s)':<12} | {'Speedup':<10}")
    print("-" * 55)
    for res in results:
        print(f"{res[0]:<12} | {res[1]:<12.4f} | {res[2]:<12.4f} | {res[3]:<10.2f}x")

if __name__ == "__main__":
    main()
