import numpy as np
class fullyconnected:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # Initialize weights with random values
        self.weights = np.random.randn(input_size, output_size)
        # Initialize biases with zeros
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):

        self.input_data = input_data
        self.output = np.dot(input_data, self.weights) + self.biases
        return self.output

    def backward(self, d_output, learning_rate):
        d_input = np.dot(d_output, self.weights.T)
        d_weights = np.dot(self.input_data.T, d_output)
        d_biases = np.sum(d_output, axis=0, keepdims=True)

        # Update parameters
        self.weights -= learning_rate * d_weights / self.input_data.shape[0]
        self.biases -= learning_rate * d_biases / self.input_data.shape[0]
        return d_input