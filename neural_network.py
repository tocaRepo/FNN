import numpy as np


# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the weights with random values
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)

        # Initialize the biases with zeros
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def feedforward(self, input_data):
        # Calculate the hidden layer's output
        self.hidden_layer_input = (
            np.dot(input_data, self.weights_input_hidden) + self.bias_hidden
        )
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)

        # Calculate the output layer's output
        self.output_layer_input = (
            np.dot(self.hidden_layer_output, self.weights_hidden_output)
            + self.bias_output
        )
        self.output_layer_output = sigmoid(self.output_layer_input)

        return self.output_layer_output

    def train(self, input_data, target_data, learning_rate):
        # Perform feedforward pass
        self.feedforward(input_data)

        # Calculate the error
        output_error = target_data - self.output_layer_output

        # Calculate the gradient for the output layer
        delta_output = output_error * sigmoid_derivative(self.output_layer_output)

        # Calculate the error in the hidden layer
        hidden_layer_error = delta_output.dot(self.weights_hidden_output.T)

        # Calculate the gradient for the hidden layer
        delta_hidden = hidden_layer_error * sigmoid_derivative(self.hidden_layer_output)

        # Update the weights and biases
        self.weights_hidden_output += (
            self.hidden_layer_output.T.dot(delta_output) * learning_rate
        )
        self.weights_input_hidden += input_data.T.dot(delta_hidden) * learning_rate
        self.bias_output += np.sum(delta_output, axis=0) * learning_rate
        self.bias_hidden += np.sum(delta_hidden, axis=0) * learning_rate

    def save_model(self, model_name):
        np.savez(
            model_name + ".npz",
            weights_input_hidden=self.weights_input_hidden,
            weights_hidden_output=self.weights_hidden_output,
            bias_hidden=self.bias_hidden,
            bias_output=self.bias_output,
        )

    def load_model(self, model_name):
        saved_model = np.load(model_name + ".npz")
        self.weights_input_hidden = saved_model["weights_input_hidden"]
        self.weights_hidden_output = saved_model["weights_hidden_output"]
        self.bias_hidden = saved_model["bias_hidden"]
        self.bias_output = saved_model["bias_output"]
