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
from em_app.currentcoils import RingCoil
from em_app import plotting
import matplotlib.pyplot as plt
from mtflib import mtf

mtf.initialize_mtf(max_order=6, max_dimension=4)

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
axis_direction = np.array([0, 0, 1])
for z_pos in ring_positions_z_vis:
    center_point = np.array([0, 0, z_pos])
    loops.append(RingCoil(1.0, solenoid_radius, 20, center_point, axis_direction))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
for loop in loops:
    loop.plot(ax)
ax.set_title("Geometry of the Solenoid Model")
plt.show()

# %% [markdown]
# ## 2. Numerical B-Field Calculation

# %%
num_rings = 100
current = 1.0

solenoid_loops = []
ring_positions_z = np.linspace(-solenoid_length / 2, solenoid_length / 2, num_rings)
axis_direction = np.array([0, 0, 1])
for z_pos in ring_positions_z:
    center_point = np.array([0, 0, z_pos])
    solenoid_loops.append(RingCoil(current, solenoid_radius, 20, center_point, axis_direction))

field_point = np.array([[0, 0, 0]])

total_B_vector = np.zeros(3, dtype=np.complex128)
for loop in solenoid_loops:
    b_vector = loop.biot_savart(field_point)[0]
    total_B_vector += b_vector.to_numpy_array()

print(f"Computed Bz from Solenoid Model: {total_B_vector[2]}")

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
