"""Build an end-to-end working deep learning model using PyTorch library -
 https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# ============================================================
# 1. LOAD DATASET
# ============================================================

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


# ============================================================
# 2. CREATE DATALOADERS
# ============================================================

batch_size = 64

train_dataloader = DataLoader(
    training_data,
    batch_size=batch_size,
    shuffle=True
)

test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False
)


# ============================================================
# 3. SELECT DEVICE
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# 4. DEFINE NEURAL NETWORK
# ============================================================

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(

            nn.Linear(28 * 28, 512),

            nn.ReLU(),

            nn.Linear(512, 512),

            nn.ReLU(),

            nn.Linear(512, 10)
        )

    def forward(self, x):

        x = self.flatten(x)

        logits = self.linear_relu_stack(x)

        return logits


# Create model
model = NeuralNetwork().to(device)

print(model)


# ============================================================
# 5. LOSS FUNCTION AND OPTIMIZER
# ============================================================

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# ============================================================
# 6. TRAINING FUNCTION
# ============================================================

def train(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):

        # Move data to device
        X = X.to(device)
        y = y.to(device)

        # Forward pass
        pred = model(X)

        # Calculate loss
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()

        # Display progress
        if batch % 100 == 0:

            loss_value = loss.item()

            current = batch * len(X)

            print(
                f"loss: {loss_value:>7f} "
                f"[{current:>5d}/{size:>5d}]"
            )


# ============================================================
# 7. TESTING FUNCTION
# ============================================================

def test(dataloader, model, loss_fn):

    model.eval()

    size = len(dataloader.dataset)

    num_batches = len(dataloader)

    test_loss = 0

    correct = 0

    with torch.no_grad():

        for X, y in dataloader:

            X = X.to(device)

            y = y.to(device)

            # Prediction
            pred = model(X)

            # Calculate loss
            test_loss += loss_fn(pred, y).item()

            # Calculate correct predictions
            correct += (
                (pred.argmax(1) == y)
                .type(torch.float)
                .sum()
                .item()
            )

    test_loss /= num_batches

    accuracy = correct / size

    print(
        f"Test Error: "
        f"Accuracy: {100 * accuracy:.1f}%, "
        f"Avg loss: {test_loss:.6f}"
    )


# ============================================================
# 8. TRAIN THE MODEL
# ============================================================

epochs = 5

for epoch in range(epochs):

    print()
    print(f"Epoch {epoch + 1}/{epochs}")

    train(
        train_dataloader,
        model,
        loss_fn,
        optimizer
    )

    test(
        test_dataloader,
        model,
        loss_fn
    )


print()
print("Training complete!")


# ============================================================
# 9. MAKE A PREDICTION
# ============================================================

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


model.eval()

x, y = test_data[0]

with torch.no_grad():

    x = x.to(device)

    pred = model(x)

    predicted_class = pred.argmax(1).item()


print()
print("Prediction:")
print("Predicted :", classes[predicted_class])
print("Actual    :", classes[y])


# ============================================================
# 10. SAVE THE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "fashion_mnist_model.pth"
)

print()
print("Model saved as: fashion_mnist_model.pth")