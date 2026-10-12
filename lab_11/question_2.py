"""Implement maxpool operation from scratch. """

import numpy as np
# Create a 32x32 input image
image = np.random.randint(0, 256, (32, 32))
# Define pooling parameters
pool_size = 2
stride = 2
# Get input dimensions
image_h, image_w = image.shape
# Calculate output dimensions
output_h = (image_h - pool_size) // stride + 1
output_w = (image_w - pool_size) // stride + 1
# Initialize output
output = np.zeros((output_h, output_w), dtype=int)
# Perform max pooling
for i in range(output_h):
    for j in range(output_w):
        region = image[
            i*stride:i*stride+pool_size,
            j*stride:j*stride+pool_size
        ]
        output[i, j] = np.max(region)
print("Input Image Shape:", image.shape)
print("Pooling Window:", (pool_size, pool_size))
print("Stride:", stride)
print("Output Feature Map Shape:", output.shape)
print("\nOutput Feature Map:\n", output)