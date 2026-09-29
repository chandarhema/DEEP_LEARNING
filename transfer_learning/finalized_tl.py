import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets,transforms,models
import os
import time
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion()

data_dir = 'data/hymenoptera_data'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
}

image_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_dataset[x], batch_size=4, shuffle=True, num_workers=4)for x in ['train', 'val']}
dataset_sizes = {x: len(image_dataset[x]) for x in ['train', 'val']}
class_names = image_dataset['train'].classes
device = (torch.accelerator.current_accelerator().type
          if torch.accelerator.is_available() else 'cpu')
print(f'{device} device is being used')

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        plt.draw()
        plt.pause(0.001)

inputs,classes = next(iter(dataloaders['train']))
imshow(torchvision.utils.make_grid(inputs),[class_names[x] for x in classes])

def train_model(model, optimizer, criterion,scheduler, num_epochs=10):
    since = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        path=os.path.join(tmpdir, 'best_model_params.pth')
        torch.save(model.state_dict(), path)
        best_acc=0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs-1}')
            print('-'*10)

            for phase in ['train', 'val']:
                model.train(True)  if phase == 'train' else model.train(False)
                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                loss = running_loss / dataset_sizes[phase]
                acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} loss: {loss:.4f} acc: {acc:.4f}')

                if phase == 'val' and acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), path)

        elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            elapsed // 60, elapsed % 60
        ))
        print('Best val Acc: {:4f}'.format(best_acc))

        model.load_state_dict(torch.load(path,weights_only=True))
    return model

def visualize_model(model,num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    plt.figure()

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size(0)):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow((inputs[j]))

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

    model.train(mode=was_training)

model_ft=models.resnet18(weights='IMAGENET1K_V1')
model_ft.fc = nn.Linear(model_ft.fc.in_features, 2)
model_ft=model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft= optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
scheduler_ft = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

model_ft=train_model(model_ft,optimizer_ft,criterion,scheduler_ft)

visualize_model(model_ft)

model_conv = models.resnet18(weights='IMAGENET1K_V1')
for param in model_conv.parameters():
    param.requires_grad = False

model_conv.fc = nn.Linear(model_conv.fc.in_features, 2)
model_conv=model_conv.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_conv= optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
scheduler_conv = torch.optim.lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
model_conv=train_model(model_conv,optimizer_conv,criterion,scheduler_conv)
visualize_model(model_conv)

def visualize_model_predictions(model,img_path):
    was_training = model.training
    model.eval()
    img=Image.open(img_path)
    img=data_transforms['val'](img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        ax=plt.subplot(2,2,1)
        ax.axis('off')
        ax.set_title(f'{class_names[preds[0].item()]}')
        imshow((img).cpu().data[0])
    model.train(mode=was_training)
visualize_model_predictions(model_conv,'data/hymenoptera_data/val/bees/72100438_73de9f17af.jpg')
plt.ioff()
plt.show()