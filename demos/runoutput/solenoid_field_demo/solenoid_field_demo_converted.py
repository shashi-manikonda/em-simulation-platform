# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # On-Axis Field of a Solenoid Demo
#
# This demo compares the numerically computed on-axis magnetic field of a model solenoid with the known analytical formula.

# %%
import numpy as np
from em_app import currentcoils, plotting
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Visualize the Geometry

# %%
solenoid_radius = 0.1
solenoid_length = 0.5
num_rings_vis = 10

loops = []
ring_positions_z_vis = np.linspace(
    -solenoid_length / 2, solenoid_length / 2, num_rings_vis
)
for z_pos in ring_positions_z_vis:
    pose = currentcoils.Pose(
        position=np.array([0, 0, z_pos]),
        orientation_axis=np.array([0, 0, 1]),
        orientation_angle=0,
    )
    loops.append(currentcoils.RingLoop(1.0, solenoid_radius, 20, pose))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
plotting._plot_loop_geometry(ax, loops)
ax.set_title("Geometry of the Solenoid Model")
plt.show()

# %% [markdown]
# ## 2. Numerical B-Field Calculation

# %%
num_rings = 100
current = 1.0

solenoid_loops = []
ring_positions_z = np.linspace(-solenoid_length / 2, solenoid_length / 2, num_rings)
for z_pos in ring_positions_z:
    pose = currentcoils.Pose(
        position=np.array([0, 0, z_pos]),
        orientation_axis=np.array([0, 0, 1]),
        orientation_angle=0,
    )
    solenoid_loops.append(currentcoils.RingLoop(current, solenoid_radius, 20, pose))

field_point = np.array([[0, 0, 0]])

total_B_numerical = np.zeros(3)
for loop in solenoid_loops:
    total_B_numerical += loop.biot_savart(field_point)[0]

print(f"Computed Bz from Solenoid Model: {total_B_numerical[2]}")

# %% [markdown]
# ## 3. Analytical Formula

# %%
mu_0 = 4 * np.pi * 1e-7
turns_per_meter = num_rings / solenoid_length
z1 = -solenoid_length / 2
z2 = solenoid_length / 2
z = 0

cos_theta_1 = (z - z1) / np.sqrt((z - z1) ** 2 + solenoid_radius**2)
cos_theta_2 = (z - z2) / np.sqrt((z - z2) ** 2 + solenoid_radius**2)
B_z_analytical = (mu_0 * turns_per_meter * current / 2) * (cos_theta_1 - cos_theta_2)

print(f"Analytical Bz for Solenoid: {B_z_analytical}")
