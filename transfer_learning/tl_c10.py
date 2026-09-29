import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time
import os
from tempfile import TemporaryDirectory


# ============================================================
# 1. DEVICE
# ============================================================

cudnn.benchmark = True
plt.ion()

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

print(f"Using {device} device")


# ============================================================
# 2. CIFAR-10 CLASSES
# ============================================================

class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

num_classes = len(class_names)

print("Number of classes:", num_classes)
print("Classes:", class_names)


# ============================================================
# 3. DATA TRANSFORMS
# ============================================================

data_transforms = {

    'train': transforms.Compose([

        # CIFAR-10 is 32 x 32
        # ResNet18 expects a larger image
        transforms.Resize((224, 224)),

        # Data augmentation
        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(10),

        # Convert image to tensor
        transforms.ToTensor(),

        # ImageNet normalization
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),

    'test': transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
}


# ============================================================
# 4. LOAD CIFAR-10
# ============================================================

data_dir = './data'


image_datasets = {

    'train': datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=data_transforms['train']
    ),

    'test': datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=data_transforms['test']
    )
}


# ============================================================
# 5. DATALOADERS
# ============================================================

dataloaders = {

    'train': torch.utils.data.DataLoader(
        image_datasets['train'],
        batch_size=64,
        shuffle=True,
        num_workers=2
    ),

    'test': torch.utils.data.DataLoader(
        image_datasets['test'],
        batch_size=64,
        shuffle=False,
        num_workers=2
    )
}


# ============================================================
# 6. DATASET SIZES
# ============================================================

dataset_sizes = {
    x: len(image_datasets[x])
    for x in ['train', 'test']
}

print("Dataset sizes:")
print(dataset_sizes)


# ============================================================
# 7. DISPLAY IMAGES
# ============================================================

def imshow(inp, title=None):

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array(
        [0.485, 0.456, 0.406]
    )

    std = np.array(
        [0.229, 0.224, 0.225]
    )

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:
        plt.title(title)

    plt.pause(0.001)


inputs, classes = next(
    iter(dataloaders['train'])
)

out = torchvision.utils.make_grid(
    inputs[:8]
)

imshow(
    out,
    title=[
        class_names[x]
        for x in classes[:8]
    ]
)


# ============================================================
# 8. TRAINING FUNCTION
# ============================================================

def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs=10
):

    since = time.time()

    with TemporaryDirectory() as tempdir:

        path = os.path.join(
            tempdir,
            'best_model_params.pt'
        )

        # Save initial model
        torch.save(
            model.state_dict(),
            path
        )

        best_acc = 0.0

        for epoch in range(num_epochs):

            print(
                f'Epoch {epoch + 1}/{num_epochs}'
            )

            print('-' * 10)

            # ================================================
            # TRAIN / TEST
            # ================================================

            for phase in ['train', 'test']:

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # ============================================
                # BATCH LOOP
                # ============================================

                for inputs, labels in dataloaders[phase]:

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Clear gradients
                    optimizer.zero_grad()

                    # Enable gradients only during training
                    with torch.set_grad_enabled(
                        phase == 'train'
                    ):

                        # Forward pass
                        outputs = model(inputs)

                        # Get predicted class
                        _, preds = torch.max(
                            outputs,
                            1
                        )

                        # Calculate loss
                        loss = criterion(
                            outputs,
                            labels
                        )

                        # Backpropagation
                        if phase == 'train':

                            loss.backward()

                            optimizer.step()

                    # Accumulate loss
                    running_loss += (
                        loss.item()
                        * inputs.size(0)
                    )

                    # Accumulate correct predictions
                    running_corrects += torch.sum(
                        preds == labels.data
                    )

                # ============================================
                # UPDATE LEARNING RATE
                # ============================================

                if phase == 'train':
                    scheduler.step()

                # ============================================
                # EPOCH LOSS
                # ============================================

                epoch_loss = (
                    running_loss
                    / dataset_sizes[phase]
                )

                # ============================================
                # EPOCH ACCURACY
                # ============================================

                epoch_acc = (
                    running_corrects.double()
                    / dataset_sizes[phase]
                )

                print(
                    f'{phase} Loss: {epoch_loss:.4f} '
                    f'Acc: {epoch_acc:.4f}'
                )

                # ============================================
                # SAVE BEST MODEL
                # ============================================

                if (
                    phase == 'test'
                    and epoch_acc > best_acc
                ):

                    best_acc = epoch_acc

                    torch.save(
                        model.state_dict(),
                        path
                    )

            print()

        # ====================================================
        # TRAINING COMPLETE
        # ====================================================

        elapsed = time.time() - since

        print(
            f'Training complete in '
            f'{elapsed // 60:.0f}m '
            f'{elapsed % 60:.0f}s'
        )

        print(
            f'Best test Acc: {best_acc:.4f}'
        )

        # Load best model
        model.load_state_dict(
            torch.load(
                path,
                weights_only=True
            )
        )

    return model


# ============================================================
# 9. VISUALIZE MODEL
# ============================================================

def visualize_model(
    model,
    num_images=6
):

    was_training = model.training

    model.eval()

    images_so_far = 0

    plt.figure(figsize=(10, 8))

    with torch.no_grad():

        for inputs, labels in dataloaders['test']:

            inputs = inputs.to(device)

            outputs = model(inputs)

            _, preds = torch.max(
                outputs,
                1
            )

            for j in range(inputs.size(0)):

                images_so_far += 1

                ax = plt.subplot(
                    num_images // 2,
                    2,
                    images_so_far
                )

                ax.axis('off')

                ax.set_title(
                    f'Predicted: '
                    f'{class_names[preds[j]]}\n'
                    f'Actual: '
                    f'{class_names[labels[j]]}'
                )

                imshow(
                    inputs.cpu()[j]
                )

                if images_so_far == num_images:

                    model.train(
                        mode=was_training
                    )

                    return

    model.train(
        mode=was_training
    )


# ============================================================
# 10. FINE-TUNING
# ============================================================

print()
print("========================================")
print("RESNET18 FINE-TUNING")
print("========================================")
print()


# Load ImageNet pretrained ResNet18

model_ft = models.resnet18(
    weights='IMAGENET1K_V1'
)


# Replace final layer

model_ft.fc = nn.Linear(
    model_ft.fc.in_features,
    num_classes
)


# Move to GPU/CPU

model_ft = model_ft.to(device)


# ============================================================
# 11. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 12. OPTIMIZER
# ============================================================

optimizer_ft = optim.SGD(
    model_ft.parameters(),
    lr=0.001,
    momentum=0.9
)


# ============================================================
# 13. LEARNING RATE SCHEDULER
# ============================================================

scheduler_ft = lr_scheduler.StepLR(
    optimizer_ft,
    step_size=7,
    gamma=0.1
)


# ============================================================
# 14. TRAIN
# ============================================================

model_ft = train_model(
    model_ft,
    criterion,
    optimizer_ft,
    scheduler_ft,
    num_epochs=10
)


# ============================================================
# 15. VISUALIZE
# ============================================================

visualize_model(
    model_ft,
    num_images=6
)


# ============================================================
# 16. FEATURE EXTRACTION
# ============================================================

print()
print("========================================")
print("RESNET18 FEATURE EXTRACTION")
print("========================================")
print()


# Load pretrained ResNet18

model_conv = models.resnet18(
    weights='IMAGENET1K_V1'
)


# ============================================================
# 17. FREEZE PRETRAINED LAYERS
# ============================================================

for param in model_conv.parameters():

    param.requires_grad = False


# ============================================================
# 18. REPLACE FINAL LAYER
# ============================================================

model_conv.fc = nn.Linear(
    model_conv.fc.in_features,
    num_classes
)


# Move model

model_conv = model_conv.to(device)


# ============================================================
# 19. OPTIMIZER
# ============================================================

optimizer_conv = optim.SGD(
    model_conv.fc.parameters(),
    lr=0.001,
    momentum=0.9
)


# ============================================================
# 20. SCHEDULER
# ============================================================

scheduler_conv = lr_scheduler.StepLR(
    optimizer_conv,
    step_size=7,
    gamma=0.1
)


# ============================================================
# 21. TRAIN FEATURE EXTRACTION MODEL
# ============================================================

model_conv = train_model(
    model_conv,
    criterion,
    optimizer_conv,
    scheduler_conv,
    num_epochs=10
)


# ============================================================
# 22. VISUALIZE
# ============================================================

visualize_model(
    model_conv,
    num_images=6
)


# ============================================================
# 23. PREDICT ONE IMAGE
# ============================================================

def visualize_model_prediction(
    model,
    dataset,
    index
):

    was_training = model.training

    model.eval()

    # Get image
    img, label = dataset[index]

    # Add batch dimension
    img_batch = img.unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():

        outputs = model(img_batch)

        _, preds = torch.max(
            outputs,
            1
        )

    predicted_class = class_names[
        preds[0].item()
    ]

    actual_class = class_names[
        label
    ]

    print(
        f'Actual: {actual_class}'
    )

    print(
        f'Predicted: {predicted_class}'
    )

    # ========================================================
    # DISPLAY IMAGE
    # ========================================================

    img_display = img.numpy().transpose(
        (1, 2, 0)
    )

    mean = np.array(
        [0.485, 0.456, 0.406]
    )

    std = np.array(
        [0.229, 0.224, 0.225]
    )

    img_display = (
        img_display * std + mean
    )

    img_display = np.clip(
        img_display,
        0,
        1
    )

    plt.figure(figsize=(5, 5))

    plt.imshow(img_display)

    plt.title(
        f'Actual: {actual_class}\n'
        f'Predicted: {predicted_class}'
    )

    plt.axis('off')

    plt.show()

    model.train(
        mode=was_training
    )


# ============================================================
# 24. TEST ONE IMAGE
# ============================================================

visualize_model_prediction(
    model_conv,
    image_datasets['test'],
    index=0
)


plt.ioff()
plt.show()