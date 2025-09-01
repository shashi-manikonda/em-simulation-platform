#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Straight Wire Magnetic Field Demo
#
# This demo showcases the `straight_wire_field` function. It compares the numerically computed magnetic field from a finite straight wire with the known analytical formula.
#
# We compute the Taylor series for the field around a point and compare its coefficients with those from the Taylor series of the analytical formula to verify the function's correctness.

# %%
import sys
import numpy as np
from mtflib import MultivariateTaylorFunction, Var, sqrt_taylor
from applications.em.straight_wire import straight_wire_field
import math
import pandas as pd

# --- Global MTF Settings ---
MultivariateTaylorFunction.initialize_mtf(max_order=4, max_dimension=3)
MultivariateTaylorFunction.set_etol(1e-12)

# --- Define Variables for MTF ---
x = Var(1)
y = Var(2)
z = Var(3)

# %% [markdown]
# ## 1. Numerical B-Field Calculation

# %%
# --- Wire and Field Point Parameters ---
start_point = [0, 0, -1]
end_point = [0, 0, 1]
current = 1.0
num_segments = 200  # More segments for better accuracy

# Define field point for Taylor expansion, e.g., around (0.1, 0, 0)
field_point_mtf = np.array([[x + 0.1, y, z]], dtype=object)

# --- Calculate the B-field numerically ---
B_numerical = straight_wire_field(
    start_point, end_point, current, field_point_mtf, num_segments=num_segments)

B_numerical_vector = B_numerical[0]  # Result has shape (1,3), so get the first row

print("Computed B-field from Straight Wire (Taylor Series Coefficients of By):")
print(B_numerical_vector[1].get_tabular_dataframe())

# %% [markdown]
# ## 2. Analytical B-Field Calculation

# %%
# Analytical formula: B = (μ₀ * I / 4π * a) * (cos(θ₁) - cos(θ₂))
# For a wire on the z-axis from z1 to z2, and a point (x,y,z), the field is in the phi direction.
# B_phi = (μ₀ * I / 4π * a) * ( (z - z1)/sqrt((z-z1)²+a²) - (z - z2)/sqrt((z-z2)²+a²) )
# where a = sqrt(x² + y²)

mu_0 = 4 * math.pi * 1e-7
z1 = -1.0
z2 = 1.0

# We are expanding around (0.1, 0, 0), so x_mtf = x + 0.1, y_mtf = y, z_mtf = z
x_mtf = x + 0.1
y_mtf = y
z_mtf = z

a_mtf = sqrt_taylor(x_mtf**2 + y_mtf**2)
cos1_mtf = (z_mtf - z1) / sqrt_taylor((z_mtf - z1)**2 + a_mtf**2)
cos2_mtf = (z_mtf - z2) / sqrt_taylor((z_mtf - z2)**2 + a_mtf**2)

B_phi_mag = (mu_0 * current / (4 * math.pi * a_mtf)) * (cos1_mtf - cos2_mtf)

# The field is in the phi direction. Bx = -B_phi * sin(phi), By = B_phi * cos(phi)
# where phi is the angle in the xy-plane. phi = atan2(y, x)
# cos(phi) = x / a, sin(phi) = y / a
cos_phi_mtf = x_mtf / a_mtf
sin_phi_mtf = y_mtf / a_mtf

Bx_analytical = -B_phi_mag * sin_phi_mtf
By_analytical = B_phi_mag * cos_phi_mtf
Bz_analytical = MultivariateTaylorFunction.from_constant(0.0)

print("Analytical B-field for Straight Wire (Taylor Series Coefficients of By):")
print(By_analytical.get_tabular_dataframe())

# %% [markdown]
# ## 3. Compare the Results

# %%
print("--- Comparison of Taylor Series Coefficients (By component) ---")
df_num = B_numerical_vector[1].get_tabular_dataframe().rename(columns={'Coefficient': 'Numerical'})
df_an = By_analytical.get_tabular_dataframe().rename(columns={'Coefficient': 'Analytical'})

comparison = pd.merge(df_num, df_an, on=['Order', 'Exponents'], how='outer').fillna(0)

comparison['RelativeError'] = (
    np.abs(comparison['Numerical'] - comparison['Analytical']) / 
    np.abs(comparison['Analytical'])
)
comparison['RelativeError'] = (
    comparison['RelativeError'].replace([np.inf, -np.inf], 0).fillna(0)
)

print(comparison[['Exponents', 'Order', 'Numerical', 'Analytical', 'RelativeError']])
