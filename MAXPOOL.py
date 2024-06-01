import numpy as np

class maxpool:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size


    def forward(self, input_data):
        self.input_data = input_data
        num_samples, input_height, input_width, num_channels = input_data.shape
        output_height = (input_height - self.pool_size) // 2 + 1
        output_width = (input_width - self.pool_size) // 2 + 1
        self.output = np.zeros((num_samples, num_channels, output_height, output_width))

        for batch_pic in range(num_samples):
            for i in range(0, input_height - self.pool_size + 1, 2):
                for j in range(0, input_width - self.pool_size + 1, 2):
                    for channel in range(num_channels):
                        portion = input_data[batch_pic, i:i + self.pool_size, j:j + self.pool_size, channel]
                        self.output[batch_pic, channel, i // 2, j // 2] = np.max(portion)
        print(self.output.shape)
        return self.output
    def backward(self, d_output, x):
        num_samples, _, _ , num_channels= self.input_data.shape
        d_input = np.zeros_like(self.input_data)

        for batch_pic in range(num_samples):
            for i in range(0, d_output.shape[1]):
                for j in range(0, d_output.shape[2]):
                    for channel in range(num_channels):
                        output_portion = self.output[batch_pic, i, j, channel]
                        input_portion = self.input_data[batch_pic, i * 2:i * 2 + self.pool_size, j * 2:j * 2 + self.pool_size, channel]
                        mask = (input_portion == output_portion)
                        d_input[batch_pic, i * 2:i * 2 + self.pool_size, j * 2:j * 2 + self.pool_size, channel] += mask * d_output[batch_pic, i, j, channel]

        return d_input
