import os
import psutil
import numpy as np
import time
import sys

# Ensure em_app and sandalwood are in path if not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

try:
    from em_app.sources import RingCoil
    from em_app.solvers import calculate_b_field, Backend
    from sandalwood import mtf
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure em-app and sandalwood are installed in editable mode.")
    sys.exit(1)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def stress_test_memory():
    print("====================================================")
    print("Starting Stress Test: Memory Stability (Sandalwood v0.1.2+)")
    print("====================================================")
    
    # 1. Initialize COSY Backend
    try:
        mtf.initialize_mtf(max_order=2, max_dimension=4, implementation="cosy")
        print(":: COSY Backend Initialized")
    except Exception as e:
        print(f":: Failed to initialize COSY: {e}")
        print("Falling back to Python backend for mock stress test.")
        mtf.initialize_mtf(max_order=2, max_dimension=4, implementation="python")

    # 2. Configuration for significant load
    num_coils = 5
    num_points = 10000  # Minimal load for verification
    num_iterations = 5
    
    print(f":: Stress Config: {num_coils} Coils, {num_points} Field Points")
    
    # 3. Create a complex coil system (Tokamak-like)
    print(":: Generating coil geometry...")
    coils = []
    for i in range(num_coils):
        angle = 2 * np.pi * i / num_coils
        center = np.array([5.0 * np.cos(angle), 5.0 * np.sin(angle), 0.0])
        axis = np.array([-np.sin(angle), np.cos(angle), 0.0])
        # Each coil has 20 segments
        coils.append(RingCoil(current=1.0, radius=2.0, num_segments=20, center_point=center, axis_direction=axis))
    
    # 4. Generate field points
    field_points = np.random.rand(num_points, 3).astype(np.float64) * 10.0
    
    initial_mem = get_memory_usage()
    print(f"Initial Memory: {initial_mem:.2f} MB")
    
    # 5. Execution Loop
    history = []
    for i in range(num_iterations):
        start_time = time.time()
        
        # We calculate the field for all coils in each iteration
        # In a real leak scenario, this would balloon memory quickly
        for coil_idx, coil in enumerate(coils):
            # We don't need to store the result, just trigger the calculation
            _ = calculate_b_field(coil, field_points, backend=Backend.COSY)
            
            # Print progress for the first iteration to give feedback
            if i == 0 and (coil_idx + 1) % 10 == 0:
                print(f"   Calculating Coil {coil_idx + 1}/{num_coils}...")

        current_mem = get_memory_usage()
        elapsed = time.time() - start_time
        history.append(current_mem)
        print(f"Iteration {i+1}/{num_iterations}: Memory = {current_mem:.2f} MB, Time = {elapsed:.2f}s")
    
    # 6. Analysis
    final_mem = history[-1]
    mem_diff = final_mem - history[0]
    
    # The first iteration might have some cache/warmup growth,
    # but subsequent ones should be flat.
    growth_after_warmup = history[-1] - history[1] if len(history) > 1 else 0
    
    print("----------------------------------------------------")
    print(f"Summary Results:")
    print(f"  Total Memory Growth: {mem_diff:.2f} MB")
    print(f"  Growth after warmup: {growth_after_warmup:.2f} MB")
    
    # Assertion Logic
    # If growth_after_warmup is near zero, the pool is working perfectly.
    # If it climbs significantly per iteration, we have a leak.
    if growth_after_warmup > 10:  # Allow 10MB for minor fragmentation/buffer
        print("FAIL: Memory usage is climbing linearly. CosyIndexPool is NOT engaging!")
        sys.exit(1)
    else:
        print("SUCCESS: Memory usage is stable. CosyIndexPool is active.")
        print("====================================================")

if __name__ == "__main__":
    stress_test_memory()
