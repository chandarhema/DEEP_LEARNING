"""Implement various update rules used to optimize the neural network.
You can use PyTorch for the following implementations.
    SGD, Momentum, AdaGrad, etc.
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
# 3. DEVICE
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

        self.network = nn.Sequential(

            nn.Linear(28 * 28, 512),

            nn.ReLU(),

            nn.Linear(512, 256),

            nn.ReLU(),

            nn.Linear(256, 10)
        )

    def forward(self, x):

        x = self.flatten(x)

        return self.network(x)


# ============================================================
# 5. TRAINING FUNCTION
# ============================================================

def train(
    dataloader,
    model,
    loss_fn,
    optimizer
):

    model.train()

    total_loss = 0

    for X, y in dataloader:

        X = X.to(device)
        y = y.to(device)

        # Forward pass
        prediction = model(X)

        # Calculate loss
        loss = loss_fn(
            prediction,
            y
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================================
# 6. TESTING FUNCTION
# ============================================================

def test(
    dataloader,
    model,
    loss_fn
):

    model.eval()

    total_loss = 0

    correct = 0

    total = 0

    with torch.no_grad():

        for X, y in dataloader:

            X = X.to(device)
            y = y.to(device)

            prediction = model(X)

            loss = loss_fn(
                prediction,
                y
            )

            total_loss += loss.item()

            predicted_class = prediction.argmax(
                dim=1
            )

            correct += (
                predicted_class == y
            ).sum().item()

            total += y.size(0)

    average_loss = (
        total_loss / len(dataloader)
    )

    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# 7. LOSS FUNCTION
# ============================================================

loss_fn = nn.CrossEntropyLoss()


# ============================================================
# 8. CREATE FOUR DIFFERENT MODELS
# ============================================================

sgd_model = NeuralNetwork().to(device)

momentum_model = NeuralNetwork().to(device)

adagrad_model = NeuralNetwork().to(device)

adam_model = NeuralNetwork().to(device)


# ============================================================
# 9. DEFINE OPTIMIZERS
# ============================================================

# ------------------------------------------------------------
# SGD
# ------------------------------------------------------------

sgd_optimizer = torch.optim.SGD(
    sgd_model.parameters(),
    lr=0.01
)


# ------------------------------------------------------------
# SGD + Momentum
# ------------------------------------------------------------

momentum_optimizer = torch.optim.SGD(
    momentum_model.parameters(),
    lr=0.01,
    momentum=0.9
)


# ------------------------------------------------------------
# AdaGrad
# ------------------------------------------------------------

adagrad_optimizer = torch.optim.Adagrad(
    adagrad_model.parameters(),
    lr=0.01
)


# ------------------------------------------------------------
# Adam
# ------------------------------------------------------------

adam_optimizer = torch.optim.Adam(
    adam_model.parameters(),
    lr=0.001
)


# ============================================================
# 10. TRAIN ALL MODELS
# ============================================================

epochs = 5

optimizers = {

    "SGD": (
        sgd_model,
        sgd_optimizer
    ),

    "Momentum": (
        momentum_model,
        momentum_optimizer
    ),

    "AdaGrad": (
        adagrad_model,
        adagrad_optimizer
    ),

    "Adam": (
        adam_model,
        adam_optimizer
    )
}


# Store results
results = {}


for name, (model, optimizer) in optimizers.items():

    print()
    print("=" * 60)
    print(f"TRAINING WITH {name}")
    print("=" * 60)

    train_losses = []

    test_accuracies = []

    for epoch in range(epochs):

        train_loss = train(
            train_dataloader,
            model,
            loss_fn,
            optimizer
        )

        test_loss, accuracy = test(
            test_dataloader,
            model,
            loss_fn
        )

        train_losses.append(train_loss)

        test_accuracies.append(accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

    results[name] = {
        "train_losses": train_losses,
        "test_accuracies": test_accuracies,
        "final_accuracy": test_accuracies[-1]
    }


# ============================================================
# 11. FINAL COMPARISON
# ============================================================

print()
print("=" * 60)
print("FINAL OPTIMIZER COMPARISON")
print("=" * 60)

for name, result in results.items():

    print(
        f"{name:<12} : "
        f"{result['final_accuracy'] * 100:.2f}%"
    )


# ============================================================
# 12. SAVE MODELS
# ============================================================

torch.save(
    sgd_model.state_dict(),
    "model_sgd.pth"
)

torch.save(
    momentum_model.state_dict(),
    "model_momentum.pth"
)

torch.save(
    adagrad_model.state_dict(),
    "model_adagrad.pth"
)

torch.save(
    adam_model.state_dict(),
    "model_adam.pth"
)

print()
print("All models saved successfully.")