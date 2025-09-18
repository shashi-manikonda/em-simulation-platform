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
# # Straight Wire Magnetic Field Demo

# %%
import numpy as np
from em_app import currentcoils

# %% [markdown]
# ## 1. Numerical B-Field Calculation

# %%
start_point = [0, 0, -1]
end_point = [0, 0, 1]
current = 1.0
num_segments = 200

wire = currentcoils.StraightWire(current, start_point, end_point, num_segments)

observation_point = np.array([[0.1, 0, 0]])
B_numerical = wire.biot_savart(observation_point)

print(f"Computed B-field: {B_numerical[0]}")

# %% [markdown]
# ## 2. Analytical B-Field Calculation

# %%
mu_0 = 4 * np.pi * 1e-7
a = 0.1
z1 = -1.0
z2 = 1.0
z = 0

cos_theta_1 = (z - z1) / np.sqrt((z - z1) ** 2 + a**2)
cos_theta_2 = (z - z2) / np.sqrt((z - z2) ** 2 + a**2)

B_phi_mag = (mu_0 * current / (4 * np.pi * a)) * (cos_theta_1 - cos_theta_2)

print(f"Analytical B-field (magnitude): {B_phi_mag}")
