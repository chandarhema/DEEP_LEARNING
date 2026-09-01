"""Implement dropout to regularize neural networks from scratch."""
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
# 4. DROPOUT FROM SCRATCH
# ============================================================

class MyDropout(nn.Module):

    def __init__(self, p=0.5):

        super().__init__()

        # Probability of dropping a neuron
        self.p = p

        if p < 0 or p > 1:
            raise ValueError(
                "Dropout probability must be between 0 and 1"
            )


    def forward(self, x):

        # During testing, do nothing
        if not self.training:
            return x

        # If p = 1, everything would be dropped
        if self.p == 1:
            return torch.zeros_like(x)

        # Create random mask
        #
        # random value >= p  -> 1
        # random value <  p  -> 0
        #
        mask = (
            torch.rand_like(x) >= self.p
        ).float()

        # Inverted dropout
        #
        # Scaling by 1/(1-p) keeps the expected
        # activation approximately unchanged.
        return x * mask / (1 - self.p)


# ============================================================
# 5. NETWORK WITHOUT DROPOUT
# ============================================================

class NetworkWithoutDropout(nn.Module):

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
# 6. NETWORK WITH DROPOUT
# ============================================================

class NetworkWithDropout(nn.Module):

    def __init__(self):

        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(

            nn.Linear(28 * 28, 512),

            nn.ReLU(),

            MyDropout(p=0.5),

            nn.Linear(512, 256),

            nn.ReLU(),

            MyDropout(p=0.5),

            nn.Linear(256, 10)
        )


    def forward(self, x):

        x = self.flatten(x)

        return self.network(x)


# ============================================================
# 7. CREATE MODELS
# ============================================================

model_without_dropout = (
    NetworkWithoutDropout().to(device)
)

model_with_dropout = (
    NetworkWithDropout().to(device)
)


# ============================================================
# 8. LOSS FUNCTION
# ============================================================

loss_fn = nn.CrossEntropyLoss()


# ============================================================
# 9. OPTIMIZERS
# ============================================================

optimizer_without_dropout = torch.optim.Adam(
    model_without_dropout.parameters(),
    lr=1e-3
)

optimizer_with_dropout = torch.optim.Adam(
    model_with_dropout.parameters(),
    lr=1e-3
)


# ============================================================
# 10. TRAINING FUNCTION
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
# 11. TESTING FUNCTION
# ============================================================

def test(
    dataloader,
    model,
    loss_fn
):

    # Evaluation mode
    #
    # Our MyDropout class will automatically
    # stop dropping neurons.
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

            predicted_class = (
                prediction.argmax(dim=1)
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
# 12. TRAIN MODEL WITHOUT DROPOUT
# ============================================================

print("\n====================================")
print("TRAINING WITHOUT DROPOUT")
print("====================================")

epochs = 5

for epoch in range(epochs):

    train_loss = train(
        train_dataloader,
        model_without_dropout,
        loss_fn,
        optimizer_without_dropout
    )

    test_loss, accuracy = test(
        test_dataloader,
        model_without_dropout,
        loss_fn
    )

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Accuracy: {accuracy * 100:.2f}%"
    )


# ============================================================
# 13. TRAIN MODEL WITH DROPOUT
# ============================================================

print("\n====================================")
print("TRAINING WITH DROPOUT")
print("====================================")

for epoch in range(epochs):

    train_loss = train(
        train_dataloader,
        model_with_dropout,
        loss_fn,
        optimizer_with_dropout
    )

    test_loss, accuracy = test(
        test_dataloader,
        model_with_dropout,
        loss_fn
    )

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Accuracy: {accuracy * 100:.2f}%"
    )


# ============================================================
# 14. FINAL COMPARISON
# ============================================================

_, accuracy_without = test(
    test_dataloader,
    model_without_dropout,
    loss_fn
)

_, accuracy_with = test(
    test_dataloader,
    model_with_dropout,
    loss_fn
)

print("\n====================================")
print("FINAL RESULTS")
print("====================================")

print(
    f"Without Dropout : "
    f"{accuracy_without * 100:.2f}%"
)

print(
    f"With Dropout    : "
    f"{accuracy_with * 100:.2f}%"
)


# ============================================================
# 15. SAVE MODELS
# ============================================================

torch.save(
    model_without_dropout.state_dict(),
    "model_without_dropout.pth"
)

torch.save(
    model_with_dropout.state_dict(),
    "model_with_dropout.pth"
)

print("\nModels saved successfully.")