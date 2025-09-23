import numpy as np
import matplotlib.pyplot as plt
from em_app.sources import RingCoil, Coil
from mtflib import mtf

mtf.initialize_mtf(max_order=6, max_dimension=4)
from em_app.plotting import plot_1d_field, plot_2d_field, plot_field_vectors_3d
from em_app.solvers import calculate_b_field
from em_app.vectors_and_fields import VectorField

class HelmholtzCoil(Coil):
    def __init__(self, current, radius, num_segments, center_point, axis_direction):
        super().__init__(current)
        self.coil1 = RingCoil(current, radius, num_segments, center_point - axis_direction * radius / 2, axis_direction)
        self.coil2 = RingCoil(current, radius, num_segments, center_point + axis_direction * radius / 2, axis_direction)
        self.segment_centers = np.concatenate([self.coil1.segment_centers, self.coil2.segment_centers])
        self.segment_lengths = np.concatenate([self.coil1.segment_lengths, self.coil2.segment_lengths])
        self.segment_directions = np.concatenate([self.coil1.segment_directions, self.coil2.segment_directions])
        self.current = current
        self.use_mtf_for_segments = self.coil1.use_mtf_for_segments

    def get_max_size(self):
        # A simple implementation for sizing the plots
        size1 = self.coil1.get_max_size()
        size2 = self.coil2.get_max_size()
        return np.maximum(size1, size2) * np.array([1,1,2])

    def get_center_point(self):
        return np.mean([self.coil1.get_center_point(), self.coil2.get_center_point()], axis=0)

    def plot(self, ax=None, **kwargs):
        self.coil1.plot(ax, **kwargs)
        self.coil2.plot(ax, **kwargs)

def main():
    # --- Part 1: Single Ring Coil Geometry ---
    print("--- Visualizing Single Ring Coil ---")
    ring_coil = RingCoil(1.0, 0.5, 20, np.array([0,0,0]), np.array([0,0,1]))

    fig1 = plt.figure(figsize=(8, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ring_coil.plot(ax=ax1)
    ax1.set_title("Single Ring Coil Geometry")
    plt.savefig("03_single_ring_geometry.png")
    plt.show()

    # --- Part 2: Helmholtz Coil Geometry ---
    print("\n--- Visualizing Helmholtz Coil ---")
    helmholtz_coil = HelmholtzCoil(1.0, 0.5, 20, np.array([0,0,0]), np.array([0,0,1]))

    fig2 = plt.figure(figsize=(8, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    helmholtz_coil.plot(ax=ax2)
    ax2.set_title("Helmholtz Coil Geometry")
    plt.savefig("03_helmholtz_geometry.png")
    plt.show()

    # --- Part 3: Field Plotting with Helmholtz Coil ---
    # 1D Plot
    print("\n--- 1D Field Plot ---")
    fig3, ax3 = plt.subplots()
    plot_1d_field(helmholtz_coil, 'z', axis='z', num_points=50, ax=ax3)
    ax3.set_title("Z-component along the z-axis of a Helmholtz Coil")
    plt.savefig("03_helmholtz_1d_plot.png")
    plt.show()

    # 2D Plot
    print("\n--- 2D Field Plot ---")
    fig4, ax4 = plt.subplots()
    plot_2d_field(helmholtz_coil, 'norm', plane='xz', num_points_a=10, num_points_b=10, ax=ax4)
    ax4.set_title("Vector Field Heatmap on the XZ-plane of a Helmholtz Coil")
    plt.savefig("03_helmholtz_2d_plot.png")
    plt.show()

    # 3D Plot
    print("\n--- 3D Field Plot ---")
    fig5 = plt.figure(figsize=(10, 8))
    ax5 = fig5.add_subplot(111, projection='3d')
    plot_field_vectors_3d(helmholtz_coil, num_points_a=5, num_points_b=5, num_points_c=5, ax=ax5)
    helmholtz_coil.plot(ax=ax5, color='r')
    ax5.set_title("3D B-field of a Helmholtz Coil")
    plt.savefig("03_helmholtz_3d_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
