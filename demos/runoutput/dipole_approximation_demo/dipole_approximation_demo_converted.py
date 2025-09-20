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
# # Magnetic Dipole Approximation Demo
#
# This demo verifies the magnetic dipole approximation.

# %%
import numpy as np
from em_app.currentcoils import RingCoil
from em_app import plotting
import matplotlib.pyplot as plt
from mtflib import mtf

mtf.initialize_mtf(max_order=6, max_dimension=4)

# %% [markdown]
# ### 1. Visualize the Geometry

# %%
ring_radius = 0.01
center_point = np.array([0, 0, 0])
axis_direction = np.array([0, 0, 1])
ring = RingCoil(1.0, ring_radius, 20, center_point, axis_direction)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ring.plot(ax)
ax.set_title("Geometry of the Current Loop")
plt.show()


# %% [markdown]
# ### 2. Function to Calculate Approximation Error

# %%
def calculate_error_at_distance(distance):
    ring_radius = 0.01
    current = 1.0
    center_point = np.array([0, 0, 0])
    axis_direction = np.array([0, 0, 1])
    loop = RingCoil(current, ring_radius, 20, center_point, axis_direction)

    field_point = np.array([[0, distance, 0]])
    B_vector = loop.biot_savart(field_point)[0]
    B_numerical = B_vector.to_numpy_array()

    mu_0_4pi = 1e-7
    area = np.pi * ring_radius**2
    m_vec = np.array([0, 0, current * area])
    r_vec = np.array([0, distance, 0])
    r_mag = np.linalg.norm(r_vec)

    term1 = 3 * np.dot(m_vec, r_vec) * r_vec / (r_mag**5)
    term2 = m_vec / (r_mag**3)
    B_analytical = mu_0_4pi * (term1 - term2)

    error = np.linalg.norm(B_numerical - B_analytical)
    return error


# %% [markdown]
# ### 3. Run Simulation Over a Range of Distances

# %%
distances = np.logspace(0, 2, 20)
errors = []
for d in distances:
    error = calculate_error_at_distance(d)
    print(f"Distance: {d:.2f}, Error: {error:.2e}")
    errors.append(error)

# %% [markdown]
# ### 4. Plot Error vs. Distance

# %%
plt.figure(figsize=(10, 6))
plt.plot(distances, errors, "o-")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Distance from loop (m)")
plt.ylabel("Approximation Error")
plt.title("Dipole Approximation Error vs. Distance")
plt.grid(True, which="both", ls="--")
plt.show()
