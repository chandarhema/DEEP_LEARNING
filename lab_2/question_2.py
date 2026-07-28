# Implement forward pass for the above two networks.
# Print activation values for each neuron at each layer.
# Print the loss value (y^).
import numpy as np

# ReLU activation function
def relu(z):
    return np.maximum(0, z)

# ======================================================
# Network 1 : Input Layer -> Output Layer
# ======================================================

print("========== NETWORK 1 ==========\n")

# Creating random input values for 4 input neurons
x = np.random.randn(4, 1)

# Creating random weights for 1 output neuron
W = np.random.randn(1, 4)

# Creating random bias
b = np.random.randn(1, 1)

# Calculating weighted sum
z1 = W @ x + b

# Applying ReLU activation
a1 = relu(z1)

# Final prediction
y_hat1 = a1

# Printing the values
print("Input Layer:")
print(x)

print("\nOutput Layer:")
print("Weighted Sum (z1):")
print(z1)

print("\nActivation Value (a1):")
print(a1)

print("\nFinal Prediction (y^):")
print(y_hat1)


# ======================================================
# Network 2 : Input -> Hidden Layer 1 -> Hidden Layer 2 -> Output
# ======================================================

print("\n\n========== NETWORK 2 ==========\n")

# Creating random input values
x = np.random.randn(4, 1)

# Hidden Layer 1 (3 neurons)
W1 = np.random.randn(3, 4)
b1 = np.random.randn(3, 1)

# Hidden Layer 2 (2 neurons)
W2 = np.random.randn(2, 3)
b2 = np.random.randn(2, 1)

# Output Layer (1 neuron)
W3 = np.random.randn(1, 2)
b3 = np.random.randn(1, 1)

# Hidden Layer 1 calculations
z1 = W1 @ x + b1
a1 = relu(z1)

# Hidden Layer 2 calculations
z2 = W2 @ a1 + b2
a2 = relu(z2)

# Output Layer calculations
z3 = W3 @ a2 + b3
a3 = relu(z3)

# Final prediction
y_hat2 = a3

# Printing the values
print("Input Layer:")
print(x)

print("\nHidden Layer 1")
print("Weighted Sum (z1):")
print(z1)

print("Activation Values (a1):")
print(a1)

print("\nHidden Layer 2")
print("Weighted Sum (z2):")
print(z2)

print("Activation Values (a2):")
print(a2)

print("\nOutput Layer")
print("Weighted Sum (z3):")
print(z3)

print("Activation Value (a3):")
print(a3)

print("\nFinal Prediction (y^):")
print(y_hat2)