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
# # Rectangular Loop Magnetic Field Demo

# %%
import numpy as np
from em_app import currentcoils, plotting
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Define Geometry and Visualize

# %%
pose = currentcoils.Pose(
    position=np.array([0.5, 0.5, 0.5]),
    orientation_axis=np.array([0, 1, 1]),
    orientation_angle=np.pi / 4,
)

rect_loop = currentcoils.RectangularLoop(
    current=1.0, width=1.0, height=1.0, num_segments_per_side=20, pose=pose
)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
plotting._plot_loop_geometry(ax, [rect_loop])
ax.set_title("Geometry of the Rectangular Loop")
plt.show()

# %% [markdown]
# ## 2. Numerical B-Field Calculation

# %%
observation_point = np.array([[0.5, 0.5, -1.0]])
B_numerical = rect_loop.biot_savart(observation_point)

print(f"Computed B-field at the observation point: {B_numerical[0]}")
