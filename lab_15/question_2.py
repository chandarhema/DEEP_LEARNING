import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# Configuration
SEED = 569
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
FEATURE_SIZE = 512
HIDDEN_SIZE = 256
EMBED_SIZE = 128
VOCAB_SIZE = 12
MAX_LENGTH = 8
EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 0.001

# Vocabulary
word_to_id = {
    "<PAD>": 0, "<START>": 1, "<END>": 2,
    "a": 3, "cat": 4, "dog": 5, "man": 6,
    "woman": 7, "runs": 8, "sits": 9,
    "on": 10, "grass": 11
}
id_to_word = {v: k for k, v in word_to_id.items()}

# Synthetic dataset
class ImageCaptionDataset(Dataset):
    def __init__(self, size=32):
        self.images = torch.rand(size, 3, IMAGE_SIZE, IMAGE_SIZE)
        self.captions = []
        examples = [
            ["<START>", "a", "cat", "sits", "on", "grass", "<END>"],
            ["<START>", "a", "dog", "runs", "on", "grass", "<END>"],
            ["<START>", "a", "man", "runs", "on", "grass", "<END>"],
            ["<START>", "a", "woman", "sits", "on", "grass", "<END>"]
        ]
        for i in range(size):
            caption = examples[i % len(examples)]
            ids = [word_to_id[w] for w in caption]
            ids += [word_to_id["<PAD>"]] * (MAX_LENGTH - len(ids))
            self.captions.append(torch.tensor(ids[:MAX_LENGTH]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.captions[idx]


# CNN image encoder
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, images):
        return self.cnn(images).flatten(1)


# LSTM image captioning model
class LSTMImageCaptioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ImageEncoder()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_SIZE, padding_idx=0)
        self.image_projection = nn.Linear(FEATURE_SIZE, EMBED_SIZE)
        self.lstm = nn.LSTM(
            input_size=EMBED_SIZE,
            hidden_size=HIDDEN_SIZE,
            batch_first=True
        )
        self.fc = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)

    def forward(self, images, captions):
        image_features = self.encoder(images)
        image_token = self.image_projection(image_features).unsqueeze(1)
        word_embeddings = self.embedding(captions[:, :-1])
        inputs = torch.cat([image_token, word_embeddings], dim=1)
        outputs, _ = self.lstm(inputs)
        return self.fc(outputs)

    def generate_caption(self, image, max_length=MAX_LENGTH):
        self.eval()
        with torch.no_grad():
            features = self.encoder(image.unsqueeze(0))
            current = self.image_projection(features).unsqueeze(1)
            state = None
            words = []

            for _ in range(max_length):
                output, state = self.lstm(current, state)
                logits = self.fc(output[:, -1, :])
                token = logits.argmax(dim=1).item()

                if token == word_to_id["<END>"]:
                    break

                if token not in (word_to_id["<PAD>"], word_to_id["<START>"]):
                    words.append(id_to_word[token])

                current = self.embedding(
                    torch.tensor([[token]], device=DEVICE)
                )

            return " ".join(words)


# Training
dataset = ImageCaptionDataset()
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = LSTMImageCaptioner().to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=word_to_id["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("LSTM Image Captioning")
print("Device:", DEVICE)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, captions in loader:
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images, captions)

        loss = criterion(
            outputs.reshape(-1, VOCAB_SIZE),
            captions.reshape(-1)
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}], "
        f"Loss: {total_loss / len(loader):.4f}"
    )

print("\nTraining completed.")
print("Example generated caption:")
print(model.generate_caption(dataset[0][0].to(DEVICE)))