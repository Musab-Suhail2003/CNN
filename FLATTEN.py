class flatten:
    def forward(self, input_data):
        self.input_shape = input_data.shape
        self.output = input_data.reshape(input_data.shape[0], -1)
        return self.output

    def backward(self, d_output, x):
        return d_output.reshape(self.input_shape)
