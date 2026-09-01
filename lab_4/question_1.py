# Implement a 1-layer (input - output layer) neural network from scratch for the following dataset.
# This includes implementing forward and backward passes from scratch.
# Print the training loss and plot it over 1000 iterations.

# x1 x2	 x3	y
# 0	  0	 1	0
# 1	  1	 1	1
# 1   0	 1	1
# 0   1	 1	0

import numpy as np
import matplotlib.pyplot as plt
# --------------------------------------------------
# Dataset
# --------------------------------------------------
X = np.array([[0, 0, 1],[1, 1, 1],[1, 0, 1],[0, 1, 1]], dtype=float)
y = np.array([[0],[1],[1],[0]],dtype=float)
# --------------------------------------------------
# Sigmoid activation function
# --------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# --------------------------------------------------
# Initialize weights and bias
# --------------------------------------------------
np.random.seed(42)
# 3 input features -> 1 output neuron
W = np.random.randn(3, 1) * 0.01
b = 0.0
# Learning rate
learning_rate = 0.1
# Number of iterations
iterations = 1000
# Store loss values
loss_history = []
# --------------------------------------------------
# Training
# --------------------------------------------------
for i in range(iterations):
    # -------------------------
    # Forward Pass
    # -------------------------
    # Weighted sum
    z = np.dot(X, W) + b
    # Activation
    y_pred = sigmoid(z)
    # -------------------------
    # Calculate Loss
    # Binary Cross Entropy
    # -------------------------
    loss = -np.mean(
        y * np.log(y_pred + 1e-8)+(1 - y) * np.log(1 - y_pred + 1e-8))
    loss_history.append(loss)
    # -------------------------
    # Backward Pass
    # -------------------------
    # Gradient of loss w.r.t. z
    dz = y_pred - y
    # Gradient of weights
    dW = np.dot(X.T, dz) / len(X)
    # Gradient of bias
    db = np.mean(dz)
    # -------------------------
    # Update weights and bias
    # -------------------------
    W = W - learning_rate * dW
    b = b - learning_rate * db
    # Print loss
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss:.6f}")
# --------------------------------------------------
# Final Results
# --------------------------------------------------
print("\nFinal Weights:")
print(W)
print("\nFinal Bias:")
print(b)
# --------------------------------------------------
# Predictions
# --------------------------------------------------
probabilities = sigmoid(np.dot(X, W) + b)
predictions = (probabilities >= 0.5).astype(int)
print("\nPredicted probabilities:")
print(probabilities)
print("\nPredicted classes:")
print(predictions)
print("\nActual classes:")
print(y.astype(int))

# --------------------------------------------------
# Plot Training Loss
# --------------------------------------------------
plt.plot(range(1, iterations + 1), loss_history)
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("Training Loss over 1000 Iterations")
plt.grid(True)
plt.show()