# Implement the forward pass using vectorized operations, i.e. W should be a matrix, x, z and a are vectors.
# The implementation should not contain any loops.
import numpy as np

# ReLU activation function
def relu(z):
    return np.maximum(0, z)

# Creating random input vector with 4 input neurons
x = np.random.randn(4, 1)

# -----------------------------
# Hidden Layer 1
# -----------------------------

# Creating random weights and bias for 3 neurons
W1 = np.random.randn(3, 4)
b1 = np.random.randn(3, 1)

# Matrix multiplication to calculate weighted sum
z1 = W1 @ x + b1

# Applying ReLU activation
a1 = relu(z1)

# -----------------------------
# Hidden Layer 2
# -----------------------------

# Creating random weights and bias for 2 neurons
W2 = np.random.randn(2, 3)
b2 = np.random.randn(2, 1)

# Matrix multiplication to calculate weighted sum
z2 = W2 @ a1 + b2

# Applying ReLU activation
a2 = relu(z2)

# -----------------------------
# Output Layer
# -----------------------------

# Creating random weights and bias for 1 output neuron
W3 = np.random.randn(1, 2)
b3 = np.random.randn(1, 1)

# Matrix multiplication to calculate weighted sum
z3 = W3 @ a2 + b3

# Applying ReLU activation
a3 = relu(z3)

# Final prediction
y_hat = a3

# -----------------------------
# Printing all values
# -----------------------------

print("Input Vector (x):")
print(x)

print("\nHidden Layer 1 Activation:")
print(a1)

print("\nHidden Layer 2 Activation:")
print(a2)

print("\nOutput Layer Activation:")
print(a3)

print("\nFinal Prediction (y^):")
print(y_hat)