import numpy as np
class Conv:
    def __init__(self, input_channels, num_filters, filter_size,  padding):
        self.padded_input = None
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.padding = padding
        # Initialize filters with random values
        self.filters = np.random.randn(num_filters, filter_size, filter_size, input_channels)
        # Initialize biases with zeros
        self.biases = np.zeros((num_filters, 1))


    def forward(self, input_data):

        self.input_data = input_data
        num_samples, input_height, input_width, input_channels= input_data.shape
        output_height = (input_height - self.filter_size + 2 * self.padding)//1 + 1
        output_width = (input_width - self.filter_size + 2 * self.padding) //1 + 1
        self.output = np.zeros((num_samples, output_height, output_width, self.num_filters))

        # Apply convolution operation
        for batch_pic in range(num_samples):
            self.padded_input = np.pad(input_data[batch_pic], ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
            for i in range(0, input_height):
                for j in range(0, input_width):
                    for filter_no in range(self.num_filters):
                        portion = self.padded_input[i:i + self.filter_size, j:j + self.filter_size, :]
                        self.output[batch_pic, i // 1, j // 1, filter_no] = np.sum(portion * self.filters[filter_no]) + self.biases[filter_no]
        return self.output

    def backward(self, d_output, learning_rate):
        num_samples = self.input_data.shape[0]
        d_input = np.zeros_like(self.padded_input)/1.0
        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        #d_input_batch = np.ones_like(d_output.shape)

        for batch_pic in range(num_samples):
            padded_input = np.pad(self.input_data[batch_pic], ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
            for i in range(self.output.shape[1]):
                for j in range(self.output.shape[2]):
                    for filter_no in range(self.num_filters):
                        portion = d_output[batch_pic, i, j, filter_no]
                        portion1 = padded_input[i:i + self.filter_size, j:j + self.filter_size, :]
                        portion = np.ones_like(portion1) * portion
                        d_filters[filter_no] += np.multiply(portion, portion1)

                        d_input[i:i + self.filter_size, j:j + self.filter_size, :] += d_output[batch_pic, i, j, filter_no] * self.filters[filter_no]
                        #d_input_batch[batch_pic, i:i + self.filter_size, j:j + self.filter_size, :] = d_input[i:i + self.filter_size, j:j + self.filter_size, :]
                        self.filters[filter_no] -= learning_rate * d_filters[filter_no] / num_samples
                        self.biases[filter_no] -= (learning_rate * d_biases[filter_no]) / num_samples
        # Update parameters
