"""Implement batch normalization from scratch. and layer normalization for training deep networks. """
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
# 2. DEVICE
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# 3. BATCH NORMALIZATION FROM SCRATCH
# ============================================================

class MyBatchNorm1d(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1):

        super().__init__()

        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = nn.Parameter(
            torch.ones(num_features)
        )

        self.beta = nn.Parameter(
            torch.zeros(num_features)
        )

        # Running statistics
        self.register_buffer(
            "running_mean",
            torch.zeros(num_features)
        )

        self.register_buffer(
            "running_var",
            torch.ones(num_features)
        )


    def forward(self, x):

        if self.training:

            # Calculate mean for each feature
            mean = x.mean(dim=0)

            # Calculate variance for each feature
            var = x.var(
                dim=0,
                unbiased=False
            )

            # Normalize
            x_normalized = (
                x - mean
            ) / torch.sqrt(
                var + self.eps
            )

            # Update running statistics
            with torch.no_grad():

                self.running_mean.mul_(
                    1 - self.momentum
                ).add_(
                    self.momentum * mean
                )

                self.running_var.mul_(
                    1 - self.momentum
                ).add_(
                    self.momentum * var
                )

        else:

            # During testing use running statistics
            x_normalized = (
                x - self.running_mean
            ) / torch.sqrt(
                self.running_var + self.eps
            )


        # Scale and shift
        return (
            self.gamma * x_normalized
            + self.beta
        )


# ============================================================
# 4. LAYER NORMALIZATION FROM SCRATCH
# ============================================================

class MyLayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-5):

        super().__init__()

        self.eps = eps

        # Learnable parameters
        self.gamma = nn.Parameter(
            torch.ones(normalized_shape)
        )

        self.beta = nn.Parameter(
            torch.zeros(normalized_shape)
        )


    def forward(self, x):

        # Mean across features of each individual sample
        mean = x.mean(
            dim=-1,
            keepdim=True
        )

        # Variance across features
        var = x.var(
            dim=-1,
            keepdim=True,
            unbiased=False
        )

        # Normalize
        x_normalized = (
            x - mean
        ) / torch.sqrt(
            var + self.eps
        )

        # Scale and shift
        return (
            self.gamma * x_normalized
            + self.beta
        )


# ============================================================
# 5. NEURAL NETWORK USING BATCH NORMALIZATION
# ============================================================

class BatchNormNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(

            nn.Linear(28 * 28, 512),

            MyBatchNorm1d(512),

            nn.ReLU(),

            nn.Linear(512, 256),

            MyBatchNorm1d(256),

            nn.ReLU(),

            nn.Linear(256, 10)
        )


    def forward(self, x):

        x = self.flatten(x)

        return self.network(x)


# ============================================================
# 6. NEURAL NETWORK USING LAYER NORMALIZATION
# ============================================================

class LayerNormNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(

            nn.Linear(28 * 28, 512),

            MyLayerNorm(512),

            nn.ReLU(),

            nn.Linear(512, 256),

            MyLayerNorm(256),

            nn.ReLU(),

            nn.Linear(256, 10)
        )


    def forward(self, x):

        x = self.flatten(x)

        return self.network(x)


# ============================================================
# 7. CREATE MODELS
# ============================================================

batch_model = BatchNormNetwork().to(device)

layer_model = LayerNormNetwork().to(device)


# ============================================================
# 8. LOSS FUNCTION
# ============================================================

loss_fn = nn.CrossEntropyLoss()


# ============================================================
# 9. OPTIMIZERS
# ============================================================

batch_optimizer = torch.optim.Adam(
    batch_model.parameters(),
    lr=1e-3
)

layer_optimizer = torch.optim.Adam(
    layer_model.parameters(),
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

        # Update parameters
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


    accuracy = correct / total

    average_loss = (
        total_loss /
        len(dataloader)
    )

    return average_loss, accuracy


# ============================================================
# 12. TRAIN BATCH NORMALIZATION MODEL
# ============================================================

print("\n==============================")
print("BATCH NORMALIZATION")
print("==============================")

epochs = 5

for epoch in range(epochs):

    train_loss = train(
        train_dataloader,
        batch_model,
        loss_fn,
        batch_optimizer
    )

    test_loss, accuracy = test(
        test_dataloader,
        batch_model,
        loss_fn
    )

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Accuracy: {accuracy * 100:.2f}%"
    )


# ============================================================
# 13. TRAIN LAYER NORMALIZATION MODEL
# ============================================================

print("\n==============================")
print("LAYER NORMALIZATION")
print("==============================")

for epoch in range(epochs):

    train_loss = train(
        train_dataloader,
        layer_model,
        loss_fn,
        layer_optimizer
    )

    test_loss, accuracy = test(
        test_dataloader,
        layer_model,
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

batch_test_loss, batch_accuracy = test(
    test_dataloader,
    batch_model,
    loss_fn
)

layer_test_loss, layer_accuracy = test(
    test_dataloader,
    layer_model,
    loss_fn
)

print("\n==============================")
print("FINAL RESULTS")
print("==============================")

print(
    f"Batch Normalization Accuracy: "
    f"{batch_accuracy * 100:.2f}%"
)

print(
    f"Layer Normalization Accuracy: "
    f"{layer_accuracy * 100:.2f}%"
)


# ============================================================
# 15. SAVE MODELS
# ============================================================

torch.save(
    batch_model.state_dict(),
    "batch_norm_model.pth"
)

torch.save(
    layer_model.state_dict(),
    "layer_norm_model.pth"
)

print("\nModels saved successfully.")