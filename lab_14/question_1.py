"""Implement Vanilla RNN from Scratch - Next-Character Prediction"""
import numpy as np
import string
import random

SEED = 569
random.seed(SEED)
np.random.seed(SEED)
ALPHABET = string.ascii_uppercase
VOCAB_SIZE = len(ALPHABET)
CHAR_TO_ID = {ch: i for i, ch in enumerate(ALPHABET)}
ID_TO_CHAR = {i: ch for ch, i in CHAR_TO_ID.items()}
HIDDEN_SIZE = 32
SEQUENCE_LENGTH = 8
LEARNING_RATE = 0.05
EPOCHS = 150
GRAD_CLIP = 5.0

def one_hot(index, size=VOCAB_SIZE):
    vector = np.zeros((size, 1))
    vector[index, 0] = 1.0
    return vector

def softmax(values):
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)

class VanillaRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.U = np.random.randn(hidden_size, input_size) * 0.08
        self.W = np.random.randn(hidden_size, hidden_size) * 0.08
        self.V = np.random.randn(output_size, hidden_size) * 0.08
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        xs, hs, probabilities = {}, {}, {}
        hs[-1] = np.zeros((self.hidden_size, 1))
        for t, char_index in enumerate(inputs):
            xs[t] = one_hot(char_index, self.input_size)
            hs[t] = np.tanh(self.U @ xs[t] + self.W @ hs[t - 1] + self.bh)
            logits = self.V @ hs[t] + self.by
            probabilities[t] = softmax(logits)
        return xs, hs, probabilities

    def calculate_loss(self, probabilities, targets):
        loss = 0.0
        for t, target in enumerate(targets):
            loss -= np.log(probabilities[t][target, 0] + 1e-12)
        return loss / len(targets)

    def train_sequence(self, inputs, targets):
        xs, hs, probabilities = self.forward(inputs)
        loss = self.calculate_loss(probabilities, targets)
        dU, dW, dV = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros((self.hidden_size, 1))
        for t in reversed(range(len(inputs))):
            dy = probabilities[t].copy()
            dy[targets[t], 0] -= 1.0
            dy /= len(targets)
            dV += dy @ hs[t].T
            dby += dy
            dh = self.V.T @ dy + dh_next
            da = dh * (1.0 - hs[t] ** 2)
            dbh += da
            dU += da @ xs[t].T
            dW += da @ hs[t - 1].T
            dh_next = self.W.T @ da
        for gradient in (dU, dW, dV, dbh, dby):
            np.clip(gradient, -GRAD_CLIP, GRAD_CLIP, out=gradient)
        self.U -= LEARNING_RATE * dU
        self.W -= LEARNING_RATE * dW
        self.V -= LEARNING_RATE * dV
        self.bh -= LEARNING_RATE * dbh
        self.by -= LEARNING_RATE * dby
        return loss

    def predict_next(self, text):
        valid_chars = [ch for ch in text.upper() if ch in CHAR_TO_ID]
        if not valid_chars:
            raise ValueError("Enter at least one letter from A to Z.")
        hidden = np.zeros((self.hidden_size, 1))
        for char in valid_chars:
            x = one_hot(CHAR_TO_ID[char], self.input_size)
            hidden = np.tanh(self.U @ x + self.W @ hidden + self.bh)
        probabilities = softmax(self.V @ hidden + self.by)
        return ID_TO_CHAR[int(np.argmax(probabilities))]

def create_dataset():
    text = ALPHABET * 4
    examples = []
    for start in range(VOCAB_SIZE):
        sequence = text[start:start + SEQUENCE_LENGTH + 1]
        if len(sequence) != SEQUENCE_LENGTH + 1:
            continue
        inputs = [CHAR_TO_ID[ch] for ch in sequence[:-1]]
        targets = [CHAR_TO_ID[ch] for ch in sequence[1:]]
        examples.append((inputs, targets))
    return examples

def main():
    print("=" * 45)
    print("VANILLA RNN FROM SCRATCH")
    print("=" * 45)
    model = VanillaRNN(VOCAB_SIZE, HIDDEN_SIZE, VOCAB_SIZE)
    dataset = create_dataset()
    print(f"Vocabulary: {VOCAB_SIZE} | Hidden neurons: {HIDDEN_SIZE} | Training sequences: {len(dataset)} | Epochs: {EPOCHS}")
    for epoch in range(1, EPOCHS + 1):
        random.shuffle(dataset)
        total_loss = 0.0
        for inputs, targets in dataset:
            total_loss += model.train_sequence(inputs, targets)
        average_loss = total_loss / len(dataset)
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {average_loss:.4f}")
    print("\nNEXT-CHARACTER PREDICTIONS")
    for sequence in ["ABCD", "MNOP", "WXYZ", "HELLO"]:
        print(f"Input: {sequence:8s} -> Predicted next character: {model.predict_next(sequence)}")
    print("\nTraining completed successfully.")

if __name__ == "__main__":
    main()