import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from em_app import RingCoil, Coil
from mtflib import mtf
# import matplotlib
# matplotlib.use('Qt5Agg') # Or 'Qt5Agg' if TkAgg doesn't work.

def main():
    """
    Demonstrates the plotting methods of the Coil class.
    """
    mtf.initialize_mtf(max_order=4, max_dimension=4)

    # Define a single ring coil
    ring_coil = RingCoil(
        current=1000.0,
        radius=0.5,
        num_segments=20,
        center_point=np.array([0, 0, 0]),
        axis_direction=np.array([0, 0, 1]),
    )

#    # 1. Plot the coil itself
#     print("Plotting the coil geometry...")
#     fig_coil = plt.figure()
#     ax_coil = fig_coil.add_subplot(111, projection="3d")
#     ring_coil.plot(ax=ax_coil, num_interpolation_points=10)
#     ax_coil.set_title("Ring Coil Geometry")
#     ax_coil.set_xlabel("X-axis")
#     ax_coil.set_ylabel("Y-axis")
#     ax_coil.set_zlabel("Z-axis")
#     plt.tight_layout()
#     plt.savefig("ring_coil_geometry.png")

    # # 2. Plot the 1D field component along the z-axis
    # print("Plotting 1D B-field component (Bz) along the Z-axis...")
    # fig_1d, ax_1d = plt.subplots()
    # ring_coil.plot_1d_field(
    #     field_component='Bz',
    #     axis='z',
    #     num_points=50,
    #     ax=ax_1d
    # )
    # ax_1d.set_title("1D B-field (Bz) along the Z-axis")
    # ax_1d.grid(True)
    # plt.tight_layout()
    # plt.savefig("1d_b_field_bz.png")

   # 3. Plot the 2D field using a quiver plot on the XZ-plane
    print("Plotting 2D B-field vectors on the XZ-plane...")
    fig_2d, ax_2d = plt.subplots()
    ring_coil.plot_2d_field(
        field_component='Bnorm',
        plane='xz',
        num_points_a=5,
        num_points_b=5,
        ax=ax_2d,
    )
    ax_2d.set_title("2D B-field Heatmap Plot (XZ-plane)")
    plt.tight_layout()
    plt.savefig("2d_b_field_xz.png")


    # # 4. Plot the 3D magnetic field vectors
    # print("Plotting 3D magnetic field vectors...")
    # fig_3d = plt.figure(figsize=(10, 8))
    # ax_3d = fig_3d.add_subplot(111, projection="3d")
    
    # # Plot the coil first for visual context
    # ring_coil.plot(ax=ax_3d, color='r')
    
    # # Plot the field vectors
    # Coil.plot_field_vectors_3d(
    #     num_points_a=7,
    #     num_points_b=7,
    #     num_points_c=7,
    #     ax=ax_3d
    # )
    # ax_3d.set_title("3D Magnetic Field Vectors with Coil")
    # plt.tight_layout()
    # plt.savefig("3d_b_field_vectors.png")

    # Show all plots
    plt.show()

if __name__ == "__main__":
    main()