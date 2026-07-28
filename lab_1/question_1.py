# Implement the following functions in Python from scratch. Do not use any library functions. ]
# You are allowed to use numpy and matplotlib. Generate 100 equally spaced values between -10 and 10.
# Call this list as  z. Implement the following functions and its derivative.
# Use class notes to find the expression for these functions.
# Use z as input and plot both the function outputs and its derivative outputs.
# Upload your code into Github and share it with me.
# Sigmoid
# Tanh
# ReLU (Rectified Linear Unit)
# Leaky ReLU
# Softmax (no need for visualization)
import numpy as np
import matplotlib.pyplot as plt

# Generate 100 equally spaced values between -10 and 10
z = np.linspace(-10, 10, 100)


# ==========================
# Sigmoid
# ==========================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# ==========================
# Tanh
# tanh(x) = (e^x - e^-x)/(e^x + e^-x)
# ==========================
def tanh(x):
    ex = np.exp(x)
    enx = np.exp(-x)
    return (ex - enx) / (ex + enx)


def tanh_derivative(x):
    t = tanh(x)
    return 1 - t * t


# ==========================
# ReLU
# ==========================
def relu(x):
    y = np.zeros(len(x))

    for i in range(len(x)):
        if x[i] > 0:
            y[i] = x[i]
        else:
            y[i] = 0

    return y


def relu_derivative(x):
    y = np.zeros(len(x))

    for i in range(len(x)):
        if x[i] > 0:
            y[i] = 1
        else:
            y[i] = 0

    return y


# ==========================
# Leaky ReLU
# ==========================
def leaky_relu(x, alpha=0.01):
    y = np.zeros(len(x))

    for i in range(len(x)):
        if x[i] > 0:
            y[i] = x[i]
        else:
            y[i] = alpha * x[i]

    return y


def leaky_relu_derivative(x, alpha=0.01):
    y = np.zeros(len(x))

    for i in range(len(x)):
        if x[i] > 0:
            y[i] = 1
        else:
            y[i] = alpha

    return y


# ==========================
# Softmax
# ==========================
def softmax(x):

    # Numerical stability
    shifted = x - np.max(x)

    exp_values = np.exp(shifted)

    total = 0
    for value in exp_values:
        total += value

    return exp_values / total


# Compute softmax output
softmax_output = softmax(z)

print("Softmax Output:")
print(softmax_output)
print("Sum of Softmax =", np.sum(softmax_output))


# ==========================
# Plotting
# ==========================
plt.figure(figsize=(12, 10))

# Sigmoid
plt.subplot(2, 2, 1)
plt.plot(z, sigmoid(z), label='Sigmoid')
plt.plot(z, sigmoid_derivative(z), '--', label='Derivative')
plt.title("Sigmoid")
plt.xlabel("z")
plt.ylabel("Output")
plt.grid(True)
plt.legend()

# Tanh
plt.subplot(2, 2, 2)
plt.plot(z, tanh(z), label='Tanh')
plt.plot(z, tanh_derivative(z), '--', label='Derivative')
plt.title("Tanh")
plt.xlabel("z")
plt.ylabel("Output")
plt.grid(True)
plt.legend()

# ReLU
plt.subplot(2, 2, 3)
plt.plot(z, relu(z), label='ReLU')
plt.plot(z, relu_derivative(z), '--', label='Derivative')
plt.title("ReLU")
plt.xlabel("z")
plt.ylabel("Output")
plt.grid(True)
plt.legend()

# Leaky ReLU
plt.subplot(2, 2, 4)
plt.plot(z, leaky_relu(z), label='Leaky ReLU')
plt.plot(z, leaky_relu_derivative(z), '--', label='Derivative')
plt.title("Leaky ReLU")
plt.xlabel("z")
plt.ylabel("Output")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()