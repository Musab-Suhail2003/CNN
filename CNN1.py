import numpy as np
from tqdm import tqdm
import CONV as Conv
import MAXPOOL as maxpool
import FLATTEN as flatten
import FULLYCONNECTED as fullcon

class CNN:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.layers = [
            Conv.Conv(input_channels=input_shape[3], num_filters=16, filter_size=3, padding=1),
            maxpool.maxpool(),
            flatten.flatten(),
            fullcon.fullyconnected(input_size=4096, output_size=num_classes)
        ]

    def forward(self, input_data):
        for i, layer in enumerate(tqdm(self.layers, desc="Forward pass")):
            input_data = layer.forward(input_data)

    def backward(self, d_output, learning_rate):
        for i, layer in enumerate(tqdm(self.layers[::-1], desc="Backwards pass")):
            d_output = layer.backward(d_output, learning_rate)
        print("epoch completed")

    def train(self, X_train, y_train, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            print(f"{epoch} / {num_epochs}")
            # Forward pass
            self.forward(X_train)
            # Calculate loss
            loss = cross_entropy_loss(softmax(self.layers[-1].output), y_train)
            # Print loss every 10 epochs
            print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
            # Backward pass
            d_output = np.zeros_like(self.layers[-1].output)
            num_samples = y_train.shape[0]
            d_output[np.arange(num_samples), y_train] = -1 / num_samples
            self.backward(d_output, learning_rate)

    def predict(self, X):
        self.forward(X)
        probabilities = softmax(self.layers[-1].output)
        return np.argmax(probabilities, axis=1), np.max(probabilities, axis=1)

# Helper functions
def softmax(logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def cross_entropy_loss(predicted_probabilities, true_labels):
    num_samples = len(true_labels)
    predicted_probabilities = np.clip(predicted_probabilities, 1e-15, 1 - 1e-15)
    loss = -np.sum(np.log(predicted_probabilities[np.arange(num_samples), true_labels])) / num_samples
    return loss

