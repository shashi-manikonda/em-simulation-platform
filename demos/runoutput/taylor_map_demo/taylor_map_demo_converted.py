# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: mtf
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TaylorMap Demo

# %% [markdown]
# ## 1. Initialize `mtflib` global parameters
#
# This is a mandatory step before using any `mtflib` functionality. We set the maximum order of the Taylor series and the number of variables.

# %%
import numpy as np
import mtflib
from mtflib import TaylorMap, MTF, MultivariateTaylorFunction

try:
    MultivariateTaylorFunction.initialize_mtf(max_order=4, max_dimension=2)
except RuntimeError:
    pass

# %% [markdown]
# ## 2. Create two `TaylorMap` objects
#
# Let's define two maps from R^2 to R^2.

# %% [markdown]
# ### Map 1: F(x,y) = [sin(x), cos(y)]

# %%
x = MultivariateTaylorFunction.from_variable(1, 2)
y = MultivariateTaylorFunction.from_variable(2, 2)
sin_x = mtflib.sin_taylor(x)
cos_y = mtflib.cos_taylor(y)
map_F = TaylorMap([sin_x, cos_y])
print(map_F)

# %% [markdown]
# ### Map 2: G(x,y) = [x + y, x - y]

# %%
map_G = TaylorMap([x + y, x - y])
print(map_G)

# %% [markdown]
# ## 3. Demonstrate Operations

# %% [markdown]
# ### Operation 1: Addition (F + G)

# %%
map_sum = map_F + map_G
print(map_sum)

# %% [markdown]
# ### Operation 2: Composition F(G(x,y))
#
# This computes `sin(x+y)` and `cos(x-y)`.

# %%
map_composed = map_F.compose(map_G)
print(map_composed)

# %% [markdown]
# ### Operation 3: Trace
#
# The trace is the sum of the diagonal elements of the Jacobian matrix's linear part. For `F(x,y) = [sin(x), cos(y)]`, the Jacobian is `[[cos(x), 0], [0, -sin(y)]]`. At (0,0), the linear part is `[[1, 0], [0, 0]]`, so the trace is 1.

# %%
trace_F = map_F.trace()
print(f"Trace of F at (0,0): {trace_F}")

# %% [markdown]
# ### Operation 4: Substitution
#
# Let's evaluate the composed map `F(G(x,y))` at `x=0.5, y=0.2`. This is equivalent to evaluating `[sin(x+y), cos(x-y)]` at the point, which is `sin(0.7)` and `cos(0.3)`.

# %%
eval_point = {1: 0.5, 2: 0.2}
result_array = map_composed.substitute(eval_point)
print(f"F(G(0.5, 0.2)) from TaylorMap: {result_array}")

# Compare with numpy to verify
numpy_result = [np.sin(0.7), np.cos(0.3)]
print(f"NumPy equivalent for comparison: {numpy_result}")

# %% [markdown]
# ## 4. Demonstrate Map Inversion

# %% [markdown]
# ### Operation 5: Inversion of a Map
#
# Here we create an invertible map `F(x,y) = [x + 0.1*y^2, y - 0.1*x^2]`, invert it, and then compose the result with the original map to verify that we get the identity map.

# %%
# Create an invertible map
x_inv = MultivariateTaylorFunction.from_variable(1, 2)
y_inv = MultivariateTaylorFunction.from_variable(2, 2)
f1_inv = x_inv + 0.1 * y_inv**2
f2_inv = y_inv - 0.1 * x_inv**2
map_to_invert = TaylorMap([f1_inv, f2_inv])
print("--- Original Map to Invert ---")
print(map_to_invert)

# Invert the map
inverted_map = map_to_invert.invert()
print("\n--- Inverted Map ---")
print(inverted_map)

# Verify by composing F and F_inv
composition = inverted_map.compose(map_to_invert)
print("\n--- Composition of F_inv o F ---")
print(composition)
print("\n(Result should be close to the identity map [x, y])")
