# Write down the observations from the plot for all the above functions in the code.
# What are the min and max values for the functions?
# Is the output of the function zero-centred?
# What happens to the gradient when the input values are too small or too big?
# What is the relationship between sigmoid and tanh?
# ==========================================================
# OBSERVATIONS
# ==========================================================

# 1. Sigmoid
# - Produces a smooth S-shaped curve.
# - Output range: 0 to 1.
# - Minimum value ≈ 0.
# - Maximum value ≈ 1.
# - Not zero-centered because outputs are always positive.
# - The derivative is maximum near x = 0 and approaches 0 for very
#   large positive or negative inputs, causing the vanishing gradient problem.

# 2. Tanh
# - Produces an S-shaped curve similar to sigmoid.
# - Output range: -1 to 1.
# - Minimum value ≈ -1.
# - Maximum value ≈ 1.
# - Zero-centered since outputs are symmetric around 0.
# - The derivative is largest near x = 0 and approaches 0 for very
#   large positive or negative inputs, also causing the vanishing gradient problem.

# 3. ReLU
# - Outputs 0 for negative inputs and increases linearly for positive inputs.
# - Output range: 0 to infinity.
# - Minimum value = 0.
# - Maximum value = No upper limit.
# - Not zero-centered.
# - Gradient is 0 for negative inputs and 1 for positive inputs.
# - Negative neurons may stop learning permanently (dying ReLU problem).

# 4. Leaky ReLU
# - Similar to ReLU but allows a small negative output for negative inputs.
# - Output range: (-infinity, infinity).
# - Minimum value = No fixed minimum.
# - Maximum value = No upper limit.
# - Not perfectly zero-centered.
# - Gradient is a small constant (e.g., 0.01) for negative inputs and
#   1 for positive inputs, reducing the dying ReLU problem.

# ==========================================================
# GENERAL OBSERVATIONS
# ==========================================================

# Minimum and Maximum Values
# Sigmoid     : Min ≈ 0,  Max ≈ 1
# Tanh        : Min ≈ -1, Max ≈ 1
# ReLU        : Min = 0,  Max = Infinity
# Leaky ReLU  : Min = -Infinity, Max = Infinity

# Zero-Centered?
# Sigmoid     : No
# Tanh        : Yes
# ReLU        : No
# Leaky ReLU  : No

# Gradient for Very Small or Very Large Inputs
# - Sigmoid and Tanh gradients become nearly zero for extreme inputs,
#   leading to the vanishing gradient problem.
# - ReLU has zero gradient for negative inputs and gradient = 1 for
#   positive inputs.
# - Leaky ReLU maintains a small non-zero gradient for negative inputs,
#   allowing neurons to continue learning.

# Relationship Between Sigmoid and Tanh
# - Tanh is a scaled and shifted version of the sigmoid function.
# - Mathematical relationship:
#       tanh(x) = 2 * sigmoid(2x) - 1
# - Tanh outputs values between -1 and 1, whereas sigmoid outputs values
#   between 0 and 1.
# - Since tanh is zero-centered, it generally converges faster than sigmoid
#   during neural network training.
# ==========================================================