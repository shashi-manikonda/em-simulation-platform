import argparse
import cProfile
import os
import pstats
import time

import matplotlib.pyplot as plt
import numpy as np
from mtflib import MultivariateTaylorFunction

from em_app.solvers import mpi_biot_savart, mpi_installed, serial_biot_savart
from em_app.sources import RingCoil

# --- MPI Setup ---
if mpi_installed:
    from mpi4py import MPI
else:
    MPI = None

# --- Global Settings ---
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
PROFILER_OUTPUT_DIR = os.path.join(script_dir, "profiling_results")
os.makedirs(PROFILER_OUTPUT_DIR, exist_ok=True)


def setup_helmholtz_coil(num_segments=20):
    """
    Sets up a Helmholtz coil configuration.
    """
    radius = 0.5
    separation = 0.5
    current = 1.0

    center1 = np.array([0, 0, -separation / 2])
    coil1 = RingCoil(
        current=current,
        radius=radius,
        num_segments=num_segments,
        center_point=center1,
        axis_direction=np.array([0, 0, 1]),
    )

    center2 = np.array([0, 0, separation / 2])
    coil2 = RingCoil(
        current=current,
        radius=radius,
        num_segments=num_segments,
        center_point=center2,
        axis_direction=np.array([0, 0, 1]),
    )

    return [coil1, coil2]


def get_aggregated_coil_parts(coils):
    """Helper to flatten coil parts for biot-savart functions."""
    all_segments = []
    all_lengths = []
    all_dirs = []
    for coil in coils:
        centers_np = np.array([c.to_numpy_array() for c in coil.segment_centers])
        dirs_np = np.array([d.to_numpy_array() for d in coil.segment_directions])
        all_segments.extend(centers_np)
        all_lengths.extend(coil.segment_lengths)
        all_dirs.extend(dirs_np)
    return all_segments, all_lengths, all_dirs


def profile_serial_calculation(coils, field_points, backend_name):
    """
    Profiles the serial Biot-Savart calculation.
    """
    print(f"--- Profiling SERIAL calculation with '{backend_name}' backend ---")
    all_segments, all_lengths, all_dirs = get_aggregated_coil_parts(coils)

    profiler = cProfile.Profile()
    profiler.enable()
    _ = serial_biot_savart(
        all_segments, all_lengths, all_dirs, field_points, backend=backend_name
    )
    profiler.disable()

    output_filename = os.path.join(
        PROFILER_OUTPUT_DIR, f"calc_serial_{backend_name}.prof"
    )
    stats_filename = os.path.join(
        PROFILER_OUTPUT_DIR, f"calc_serial_{backend_name}.txt"
    )
    profiler.dump_stats(output_filename)
    with open(stats_filename, "w") as f:
        ps = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
        ps.print_stats()
    print(f"Profiling data saved to {output_filename} and {stats_filename}")


def profile_mpi_calculation(coils, field_points):
    """
    Profiles the MPI Biot-Savart calculation.
    """
    if not mpi_installed:
        print("MPI not installed. Skipping MPI profiling.")
        return

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("--- Profiling MPI calculation (backend is not configurable) ---")

    all_segments, all_lengths, all_dirs = get_aggregated_coil_parts(coils)

    # Synchronize before starting the timer
    comm.Barrier()

    profiler = cProfile.Profile()
    profiler.enable()
    _ = mpi_biot_savart(all_segments, all_lengths, all_dirs, field_points)
    profiler.disable()

    # Gather profiling stats on rank 0
    # Note: This profiles each process, but for the report we focus on rank 0's view
    output_filename = os.path.join(PROFILER_OUTPUT_DIR, f"calc_mpi_rank_{rank}.prof")
    stats_filename = os.path.join(PROFILER_OUTPUT_DIR, f"calc_mpi_rank_{rank}.txt")
    profiler.dump_stats(output_filename)

    if rank == 0:
        with open(stats_filename, "w") as f:
            ps = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
            ps.print_stats()
        print(f"Profiling data for all ranks saved in '{PROFILER_OUTPUT_DIR}'")


def profile_plotting(coils, field_points, backend_name):
    """
    Profiles the plotting functions, separating calculation from drawing.
    """
    print(f"--- Profiling PLOTTING with '{backend_name}' backend ---")
    all_segments, all_lengths, all_dirs = get_aggregated_coil_parts(coils)

    # 1. Calculation Phase (not profiled)
    print("Calculating B-field data first...")
    start_time = time.time()
    # The plotting functions call biot_savart point-by-point, so we replicate that.
    B_vectors = np.zeros((len(field_points), 3))
    for i, point in enumerate(field_points):
        B_contrib_mtf = serial_biot_savart(
            all_segments,
            all_lengths,
            all_dirs,
            np.array([point]),
            order=0,
            backend=backend_name,
        )
        # Use the safe way to extract the constant coefficient (order 0 value) and convert to scalar
        B_vectors[i] = np.array([
            b.extract_coefficient(tuple([0] * b.dimension)).item()
            for b in B_contrib_mtf[0]
        ])
    calc_time = time.time() - start_time
    print(f"Calculation finished in {calc_time:.4f} seconds.")

    # 2. Plotting Phase (profiled)
    print("Profiling plotting functions...")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    profiler = cProfile.Profile()
    profiler.enable()

    # Profile the actual matplotlib calls
    for coil in coils:
        coil.plot(ax)
    ax.plot(field_points[:, 0], field_points[:, 1], field_points[:, 2], "r--")

    # A simplified quiver plot for profiling purposes
    ax.quiver(
        field_points[:, 0],
        field_points[:, 1],
        field_points[:, 2],
        B_vectors[:, 0],
        B_vectors[:, 1],
        B_vectors[:, 2],
        length=0.1,
        normalize=True,
    )

    profiler.disable()
    plt.close(fig)  # Prevent plot from showing

    output_filename = os.path.join(PROFILER_OUTPUT_DIR, f"plotting_{backend_name}.prof")
    stats_filename = os.path.join(PROFILER_OUTPUT_DIR, f"plotting_{backend_name}.txt")
    profiler.dump_stats(output_filename)
    with open(stats_filename, "w") as f:
        ps = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
        ps.print_stats()
    print(f"Profiling data saved to {output_filename} and {stats_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Performance profiling for mtflib EM demos."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Serial Calculation Parser ---
    parser_calc_serial = subparsers.add_parser(
        "calc-serial", help="Profile the serial calculation."
    )
    parser_calc_serial.add_argument(
        "--backend",
        type=str,
        choices=["python", "cpp", "c", "cpp_v2"],
        required=True,
    )
    parser_calc_serial.add_argument("--num_points", type=int, default=100)
    parser_calc_serial.add_argument("--num_segments", type=int, default=20)

    # --- MPI Calculation Parser ---
    parser_calc_mpi = subparsers.add_parser(
        "calc-mpi", help="Profile the MPI calculation."
    )
    parser_calc_mpi.add_argument("--num_points", type=int, default=100)
    parser_calc_mpi.add_argument("--num_segments", type=int, default=20)

    # --- Plotting Parser ---
    parser_plot = subparsers.add_parser("plot", help="Profile the plotting functions.")
    parser_plot.add_argument(
        "--backend",
        type=str,
        choices=["python", "cpp", "c", "cpp_v2"],
        required=True,
    )
    parser_plot.add_argument(
        "--num_points", type=int, default=50
    )  # Plotting is slow, use fewer points
    parser_plot.add_argument("--num_segments", type=int, default=20)

    args = parser.parse_args()

    # --- Initialize MTF ---
    MultivariateTaylorFunction.initialize_mtf(max_order=6, max_dimension=4)

    # In MPI, ensure all processes wait for file changes before proceeding
    if mpi_installed and "OMPI_COMM_WORLD_RANK" in os.environ:
        MPI.COMM_WORLD.Barrier()

    # --- Setup Geometry ---
    coils = setup_helmholtz_coil(num_segments=args.num_segments)
    field_points = np.linspace([0, 0, -1], [0, 0, 1], args.num_points)

    # --- Execute Command ---
    if args.command == "calc-serial":
        profile_serial_calculation(coils, field_points, args.backend)
    elif args.command == "calc-mpi":
        profile_mpi_calculation(coils, field_points)
    elif args.command == "plot":
        profile_plotting(coils, field_points, args.backend)


if __name__ == "__main__":
    main()
