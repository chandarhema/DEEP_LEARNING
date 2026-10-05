"""Using the PathMNIST dataset to build and evaluate deep learning models that classify colorectal histology
image patches into one of nine tissue classes.Pathmnist provides RGB images of size 28*28 pixels with training
validation and test splits (20000 / 3000 / 3000 images).Use these supplied splits;do not merge or resplit them .
You can use data=np.load("train_20000.npz") to load the images.You need to construct a deep CNN of your choice
to maximize the performance. However it is essential that there are atleast two CONV layers.Print the loss and
training accuracy periodically after each epoch .You need to train the model for at least 3 epochs.Use the trained
model to compute the test accuracy across the test set and print the test accuracy
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# =========================================================
# 1. CONFIGURATION
# =========================================================
DATA_DIR = "dl_dataset"
TRAIN_FILE = os.path.join(DATA_DIR, "train_20000.npz")
VAL_FILE = os.path.join(DATA_DIR, "val_3000.npz")
TEST_FILE = os.path.join(DATA_DIR, "test_3000.npz")
BATCH_SIZE = 128
EPOCHS = 6
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =========================================================
# 2. LOAD THE SUPPLIED DATASETS
# =========================================================
train_data = np.load(TRAIN_FILE)
val_data = np.load(VAL_FILE)
test_data = np.load(TEST_FILE)

print("\nFiles loaded:")
print("Train keys:", train_data.files)
print("Validation keys:", val_data.files)
print("Test keys:", test_data.files)
# Load images and labels
X_train = train_data["images"]
y_train = train_data["labels"]
X_val = val_data["images"]
y_val = val_data["labels"]
X_test = test_data["images"]
y_test = test_data["labels"]
print("\nOriginal dataset shapes:")
print("Training   :", X_train.shape, y_train.shape)
print("Validation :", X_val.shape, y_val.shape)
print("Test       :", X_test.shape, y_test.shape)
# =========================================================
# 3. PREPROCESS THE DATA
# =========================================================
# Convert images to float32
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)
# Normalize pixel values to [0, 1]
if X_train.max() > 1.0:
    X_train /= 255.0
    X_val /= 255.0
    X_test /= 255.0
# ---------------------------------------------------------
# Convert from:
# (N, H, W, C)
# to PyTorch format:
# (N, C, H, W)
# ---------------------------------------------------------
if X_train.ndim == 4 and X_train.shape[-1] == 3:
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_val = np.transpose(X_val, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))
elif X_train.ndim == 4 and X_train.shape[1] == 3:
    pass
else:
    raise ValueError("Unexpected image shape. Expected ""(N,H,W,3) or (N,3,H,W).")
# ---------------------------------------------------------
# Convert labels to 1-dimensional arrays
# ---------------------------------------------------------
y_train = np.asarray(y_train).reshape(-1)
y_val = np.asarray(y_val).reshape(-1)
y_test = np.asarray(y_test).reshape(-1)
# ---------------------------------------------------------
# Convert NumPy arrays to PyTorch tensors
# ---------------------------------------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print("\nPreprocessed shapes:")
print("Training   :", X_train.shape, y_train.shape)
print("Validation :", X_val.shape, y_val.shape)
print("Test       :", X_test.shape, y_test.shape)

# =========================================================
# 4. CREATE DATASETS AND DATALOADERS
# =========================================================
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
print("\nNumber of samples:")
print("Training   :", len(train_dataset))
print("Validation :", len(val_dataset))
print("Test       :", len(test_dataset))
# =========================================================
# 5. DEFINE DEEP CNN
# =========================================================
class DeepCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(DeepCNN, self).__init__()
        self.features = nn.Sequential(
            # -------------------------------------------------
            # CONVOLUTION BLOCK 1
            # -------------------------------------------------
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.10),
            # -------------------------------------------------
            # CONVOLUTION BLOCK 2
            # -------------------------------------------------
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.15),
            # -------------------------------------------------
            # CONVOLUTION BLOCK 3
            # -------------------------------------------------
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.20),
            # -------------------------------------------------
            # CONVOLUTION BLOCK 4
            # -------------------------------------------------
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25)
        )
        # -----------------------------------------------------
        # ADAPTIVE POOLING
            # This converts the feature map to:256 x 1 x 1
            # Therefore the Linear layer always receives 256 features regardless of whether the input is 64x64
            # or another compatible image size.
        # -----------------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # -----------------------------------------------------
        # FULLY CONNECTED CLASSIFIER
        # -----------------------------------------------------
        self.classifier = nn.Sequential(nn.Flatten(),nn.Linear(256, 128),nn.ReLU(inplace=True),nn.Dropout(0.5),
        nn.Linear(128, num_classes))
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

# =========================================================
# 6. CREATE MODEL
# =========================================================
model = DeepCNN(num_classes=9)
model = model.to(DEVICE)
print("\nModel:")
print(model)

# =========================================================
# 7. LOSS FUNCTION AND OPTIMIZER
# =========================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-4)
# Learning-rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=0.5,patience=2)

# =========================================================
# 8. TRAINING
# =========================================================
print("\nStarting training...")
print("=" * 75)
best_val_accuracy = 0.0
for epoch in range(EPOCHS):
    # -----------------------------------------------------
    # TRAINING MODEL
    # -----------------------------------------------------
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # -----------------------------------------------------
    # TRAINING BATCHES
    # -----------------------------------------------------
    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        # Clear gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        # Calculate loss
        loss = criterion(outputs, labels)
        # Backpropagation
        loss.backward()
        # Update parameters
        optimizer.step()
        # -------------------------------------------------
        # Statistics
        # -------------------------------------------------
        running_loss += loss.item() * images.size(0)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    # -----------------------------------------------------
    # EPOCH TRAINING RESULTS
    # -----------------------------------------------------
    epoch_loss = running_loss / total
    train_accuracy = 100.0 * correct / total
    # =====================================================
    # VALIDATION
    # =====================================================
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_accuracy = 100.0 * val_correct / val_total
    # -----------------------------------------------------
    # UPDATE LEARNING RATE
    # -----------------------------------------------------
    scheduler.step(val_accuracy)
    # -----------------------------------------------------
    # SAVE BEST MODEL
    # -----------------------------------------------------
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(),"best_pathmnist_cnn.pth")
    # =====================================================
    # PRINT RESULTS AFTER EVERY EPOCH
    # =====================================================
    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
        f"Loss: {epoch_loss:.4f} | "
        f"Training Accuracy: {train_accuracy:.2f}% | "
        f"Validation Accuracy: {val_accuracy:.2f}% | "
        f"LR: {current_lr:.6f}")

# =========================================================
# 9. LOAD BEST MODEL
# =========================================================
print("\nTraining completed.")
model.load_state_dict(torch.load("best_pathmnist_cnn.pth",map_location=DEVICE))
model.eval()

# =========================================================
# 10. TEST THE TRAINED MODEL
# =========================================================
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        # Forward pass
        outputs = model(images)
        # Predicted class
        predicted = torch.argmax(outputs, dim=1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
# Calculate test accuracy
test_accuracy = 100.0 * test_correct / test_total

# =========================================================
# 11. FINAL RESULTS
# =========================================================
print("\n")
print("=" * 75)
print("FINAL RESULTS")
print("=" * 75)
print(f"Training samples   : {len(train_dataset)}")
print(f"Validation samples : {len(val_dataset)}")
print(f"Test samples       : {test_total}")
print(f"Best Validation Accuracy : {best_val_accuracy:.2f}%")
print(f"Correct Test Predictions : {test_correct}")
print(f"Test Accuracy            : {test_accuracy:.2f}%")
print("=" * 75)
print("\nModel saved as: best_pathmnist_cnn.pth")


"""FINAL RESULTS
===========================================================================
Training samples         : 20000
Validation samples       : 3000
Test samples             : 3000
Best Validation Accuracy : 78.30%
Correct Test Predictions : 2395
Test Accuracy            : 79.83%
===========================================================================
if we increase the no of epochs the training accuracy will go down according to some point
but after that it will again start increasing because of the vanishing gradient problem
"""