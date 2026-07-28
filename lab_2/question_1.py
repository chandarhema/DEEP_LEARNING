
# Consider the following two networks.
# W is a matrix, x is a vector, z is a vector, and a is a vector.
# y^ is a scalar and a final prediction. Initialize x, w randomly, z is a dot product of x and w, a is ReLU(z).
# Initialize X and W randomly. Every neuron has a bias term.

import numpy as np                          # Import NumPy for matrix and vector operations

# -------------------------------
# Define the ReLU activation function
# -------------------------------
def relu(z):
    # ReLU returns the positive value if z > 0, otherwise returns 0
    return np.maximum(0, z)

# -------------------------------
# Initialize the input vector
# -------------------------------
# Create a random input vector with 4 features (4x1)
x = np.random.randn(4, 1)

# -------------------------------
# Initialize the weight matrix
# -------------------------------
# Since there is one output neuron and four input neurons,
# the weight matrix size is (1 x 4)
W = np.random.randn(1, 4)

# -------------------------------
# Initialize the bias
# -------------------------------
# Every neuron has one bias value.
# Since there is one output neuron, the bias size is (1 x 1)
b = np.random.randn(1, 1)

# -------------------------------
# Compute the weighted sum
# -------------------------------
# Matrix multiplication between W and x
# Then add the bias term
# Formula: z = Wx + b
z = W @ x + b

# -------------------------------
# Apply the ReLU activation
# -------------------------------
# Formula: a = ReLU(z)
a = relu(z)

# -------------------------------
# Final prediction
# -------------------------------
# Since there is only one output neuron,
# its activation is the final prediction (ŷ)
y_hat = a

# -------------------------------
# Print all intermediate values
# -------------------------------

print("Input Vector (x):")
print(x)

print("\nWeight Matrix (W):")
print(W)

print("\nBias (b):")
print(b)

print("\nWeighted Sum (z = Wx + b):")
print(z)

print("\nActivation (a = ReLU(z)):")
print(a)

print("\nFinal Prediction (ŷ):")
print(y_hat)