import os
import psutil
import time
import numpy as np
import gc
from em_app.solvers import calculate_b_field
from em_app.sources import RingCoil
from sandalwood import mtf

# Initialize MTF
# Max dimension 4 is standard for these sims, order 1 is sufficient for standard field calculation
mtf.initialize_mtf(max_order=1, max_dimension=4)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def run_stress_test():
    print(f"Initial Memory: {get_memory_usage():.2f} MB")

    # 1. Create coils (Tokamak-like torus)
    # A full tokamak might have 18-24 TF coils, let's use 50 to stress it properly.
    num_coils = 50
    major_radius = 5.0
    minor_radius = 1.0
    current = 1000.0
    coils = []

    print(f"Creating {num_coils} coils in toroidal arrangement...")
    for i in range(num_coils):
        angle = 2 * np.pi * i / num_coils
        center = np.array([major_radius * np.cos(angle), major_radius * np.sin(angle), 0])
        # Axis tangent to the circle: (-sin, cos, 0)
        axis = np.array([-np.sin(angle), np.cos(angle), 0])
        
        coil = RingCoil(
            current=current,
            radius=minor_radius,
            num_segments=20, 
            center_point=center,
            axis_direction=axis
        )
        coils.append(coil)

    print(f"Memory after coil creation: {get_memory_usage():.2f} MB")

    # 2. Create Grid (10^6 points)
    # Using 100x100x100 grid
    print("Creating 1,000,000 field points...")
    N = 100
    x = np.linspace(-6, 6, N)
    y = np.linspace(-6, 6, N)
    z = np.linspace(-2, 2, N)
    X, Y, Z = np.meshgrid(x, y, z)
    field_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
    assert field_points.shape[0] == 1000000
    
    print(f"Memory after grid creation: {get_memory_usage():.2f} MB")

    # 3. Compute Field
    start_time = time.time()
    total_field = None
    
    print("Computing B-field and summing contributions...")
    initial_compute_mem = get_memory_usage()
    
    # Store memory readings to check for linearity (instability)
    mem_readings = []

    for i, coil in enumerate(coils):
        # Calculate field for this coil
        b_field = calculate_b_field(coil, field_points)
        
        # Accumulate
        if total_field is None:
            total_field = b_field
        else:
            total_field = total_field + b_field
            
        if i % 5 == 0 or i == num_coils - 1:
            current_mem = get_memory_usage()
            mem_readings.append(current_mem)
            print(f"  Processed {i+1}/{num_coils} coils. Memory: {current_mem:.2f} MB")
            
    end_time = time.time()
    print(f"Composition finished in {end_time - start_time:.2f} seconds.")
    print(f"Final Memory: {get_memory_usage():.2f} MB")
    
    mem_growth = get_memory_usage() - initial_compute_mem
    print(f"Memory growth during computation phase: {mem_growth:.2f} MB")
    
    # Simple check: If growth is huge (e.g. > 1GB for 50 coils accumulation), that's suspicious,
    # but exact threshold depends on implementation.
    # We are mainly looking for 'flat' profile after initialization.
    # The first few might increase due to broadcasting/initial allocations, but it should level off.
    
    # 4. Cleanup
    del total_field
    del coils
    gc.collect()
    print(f"Memory after cleanup: {get_memory_usage():.2f} MB")

if __name__ == "__main__":
    run_stress_test()
