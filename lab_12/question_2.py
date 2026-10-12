"""Implement CNN using PyTorch for image classification using cifar10 dataset -
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html .
Plot train error vs increasing number of layers.
After some point, the training error increases with the number of layers.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Use a maximum of 8 CPU threads
torch.set_num_threads(8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

# Download CIFAR-10 dataset
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2
)


# Define CNN with variable depth
class CNN(nn.Module):

    def __init__(self, num_layers):
        super().__init__()

        layers = []
        in_channels = 3

        for i in range(num_layers):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    32,
                    kernel_size=3,
                    padding=1
                )
            )
            layers.append(nn.ReLU())
            in_channels = 32

        self.features = nn.Sequential(*layers)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)

        return x


# Number of convolutional layers to test
depths = [1, 2, 3, 4, 5, 6]

train_errors = []
epochs = 5

# Train CNNs with different depths
for depth in depths:

    print(f"\nTraining CNN with {depth} convolutional layers")

    model = CNN(depth).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Training
    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {running_loss / len(train_loader):.4f}"
        )

    # Calculate training error
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = 100 * correct / total
    error = 100 - accuracy

    train_errors.append(error)

    print(f"Training Accuracy: {accuracy:.2f}%")
    print(f"Training Error: {error:.2f}%")


# Plot training error against network depth
plt.figure(figsize=(8, 5))

plt.plot(
    depths,
    train_errors,
    marker="o"
)

plt.xlabel("Number of Convolutional Layers")
plt.ylabel("Training Error (%)")
plt.title("CNN Depth vs Training Error on CIFAR-10")

plt.xticks(depths)
plt.grid(True)
plt.tight_layout()
plt.savefig("cifar10_depth_vs_training_error.png", dpi=300)
plt.show()

print("\nResults:")
for depth, error in zip(depths, train_errors):
    print(f"Layers: {depth}, Training Error: {error:.2f}%")