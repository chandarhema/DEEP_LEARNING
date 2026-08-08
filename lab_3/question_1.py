# Implement backward pass for the above two networks.
# Print the gradient values for each neuron in each layer.
import numpy as np

# -----------------------
# Activation
# -----------------------
def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(a):
    return a*(1-a)

# -----------------------
# Input
# -----------------------
X = np.array([[1.0],
              [2.0],
              [3.0],
              [4.0]])

Y = np.array([[1]])

# weights
W = np.array([[0.2, -0.3, 0.4, 0.1]])

b = np.array([[0.5]])

# -----------------------
# Forward Pass
# -----------------------
Z = W @ X + b
A = sigmoid(Z)

loss = 0.5*(Y-A)**2

print("Prediction =",A)
print("Loss =",loss)

# -----------------------
# Backward Pass
# -----------------------

dL_dA = A - Y

dA_dZ = sigmoid_derivative(A)

dZ = dL_dA * dA_dZ

dW = dZ @ X.T

db = dZ

dX = W.T @ dZ

print("\nGradients")
print("dz =",dZ)
print("dW =")
print(dW)
print("db =",db)
print("dX =")
print(dX)