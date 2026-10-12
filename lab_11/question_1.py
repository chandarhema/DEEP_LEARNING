"""Implement convolution operations from scratch.
Assume a 3x3 kernel and apply it on an input image of 32x32."""

import numpy as np
# Create a 32x32 input image
image = np.random.randint(0, 256, (32, 32))
# Define a 3x3 kernel
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]])
# Get image and kernel dimensions
image_h, image_w = image.shape
kernel_h, kernel_w = kernel.shape
# Calculate output dimensions (valid convolution)
output_h = image_h - kernel_h + 1
output_w = image_w - kernel_w + 1
# Initialize output
output = np.zeros((output_h, output_w))
# Perform convolution operation
for i in range(output_h):
    for j in range(output_w):
        region = image[i:i+kernel_h, j:j+kernel_w]
        output[i, j] = np.sum(region * kernel)
print("Input Image Shape:", image.shape)
print("Kernel Shape:", kernel.shape)
print("Output Feature Map Shape:", output.shape)
print("\nOutput Feature Map:\n", output)