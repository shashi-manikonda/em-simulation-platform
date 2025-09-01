#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # On-Axis Field of a Solenoid Demo
#
# This demo compares the numerically computed on-axis magnetic field of a model solenoid with the known analytical formula.
#
# A solenoid is modeled as a series of discrete current rings. The total magnetic field is the sum of the fields from each ring. We compute the Taylor series for this field along the solenoid's axis and compare its coefficients with the coefficients from the Taylor series of the analytical formula.

# %%
import sys
import numpy as np
from mtflib import MultivariateTaylorFunction, Var, integrate, sqrt_taylor
from applications.em.biot_savart import serial_biot_savart
from applications.em.current_ring import current_ring
import math
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# %% [markdown]
# ## 1. Visualize the Geometry

# %%
# --- Solenoid Parameters for visualization ---
solenoid_radius = 0.1
solenoid_length = 0.5
num_rings_vis = 10  # Just a few rings for a clear visual
observation_point = [0, 0, 0]  # At the center of the solenoid

# --- Visualize the Setup ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
theta = np.linspace(0, 2 * np.pi, 100)

# Plot each ring of the solenoid
ring_positions_z_vis = np.linspace(-solenoid_length / 2, solenoid_length / 2, num_rings_vis)
for z_pos in ring_positions_z_vis:
    loop_x = solenoid_radius * np.cos(theta)
    loop_y = solenoid_radius * np.sin(theta)
    ax.plot(loop_x, loop_y, z_pos, 'b-')

# Add an arrow for current direction on the middle ring
ax.quiver(solenoid_radius, 0, 0, 0, 1, 0, 
          length=0.05, normalize=True, color='b', arrow_length_ratio=0.4)

# Plot the observation point
ax.scatter(observation_point[0], observation_point[1], observation_point[2], 
           c='r', marker='x', s=100, label='Observation Point')

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Geometry of the Solenoid Model')
ax.legend()
ax.axis('equal')
plt.show()

# %% [markdown]
# ## 2. Numerical B-Field Calculation

# %%
# --- MTF Setup ---
MultivariateTaylorFunction.initialize_mtf(max_order=6, max_dimension=4)
MultivariateTaylorFunction.set_etol(1e-12)
z = Var(3)

# --- Solenoid Parameters ---
solenoid_radius = 0.1
solenoid_length = 0.5
num_rings = 500
current = 1.0
turns_per_meter = num_rings / solenoid_length

# --- Field Point on the axis ---
field_point_mtf = np.array([[0, 0, z]], dtype=object)

# --- Calculate B-field by summing contributions from each ring ---
total_B_numerical = np.array([
    MultivariateTaylorFunction.from_constant(0.0) for _ in range(3)
], dtype=object)

ring_positions_z = np.linspace(-solenoid_length / 2, solenoid_length / 2, num_rings)
for z_pos in ring_positions_z:
    segment_mtfs, element_lengths, direction_vectors = current_ring(
        solenoid_radius, 20, np.array([0, 0, z_pos]), np.array([0, 0, 1]))
    B_ring_with_u = serial_biot_savart(
        segment_mtfs, element_lengths, direction_vectors, field_point_mtf)
    B_ring = [integrate(bfld, 4, -1, 1) for bfld in B_ring_with_u[0]]
    total_B_numerical += B_ring

B_z_numerical = total_B_numerical[2]
print("Computed Bz from Solenoid Model (Taylor Series Coefficients):")
print(B_z_numerical.get_tabular_dataframe())

# %% [markdown]
# ## 3. Implement the Analytical Formula

# %%
mu_0 = 4 * math.pi * 1e-7
z1 = -solenoid_length / 2
z2 = solenoid_length / 2

cos_theta_1 = (z - z1) / sqrt_taylor((z - z1)**2 + solenoid_radius**2)
cos_theta_2 = (z - z2) / sqrt_taylor((z - z2)**2 + solenoid_radius**2)
B_z_analytical = (mu_0 * turns_per_meter * current / 2) * (cos_theta_1 - cos_theta_2)

print("Analytical Bz for Solenoid (Taylor Series Coefficients):")
print(B_z_analytical.get_tabular_dataframe())

# %% [markdown]
# ## 4. Compare the Results

# %%
print("--- Comparison of Taylor Series Coefficients (Bz component) ---")
df_num = B_z_numerical.get_tabular_dataframe().rename(columns={'Coefficient': 'Numerical'})
df_an = B_z_analytical.get_tabular_dataframe().rename(columns={'Coefficient': 'Analytical'})

comparison = pd.merge(df_num, df_an, on=['Order', 'Exponents'], how='outer').fillna(0)
comparison['RelativeError'] = (
    np.abs(comparison['Numerical'] - comparison['Analytical']) / 
    np.abs(comparison['Analytical'])
)
comparison['RelativeError'] = (
    comparison['RelativeError'].replace([np.inf, -np.inf], 0).fillna(0)
)

print(comparison[['Exponents', 'Order', 'Numerical', 'Analytical', 'RelativeError']])
