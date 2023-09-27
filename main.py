import numpy as np
from neural_network import NeuralNetwork
import os

# Example usage
if __name__ == "__main__":
    # Define the neural network
    input_size = 2
    hidden_size = 4
    output_size = 1
    neural_network = NeuralNetwork(input_size, hidden_size, output_size)
    
    # Define some training data
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    target_data = np.array([[0], [1], [1], [0]])
    MODEL_NAME="xor_model"
    # Check if the file exists
    if os.path.isfile(MODEL_NAME+'.npz'):
        print("the model exists, we just make the prediction")
        # Load the model and continue training or make predictions
        neural_network.load_model(MODEL_NAME)
    else:
        print("The file 'trained_model.npz' does not exist. You can start training a new model.")
        # Train the neural network
        num_epochs = 10000
        learning_rate = 0.1
        for epoch in range(num_epochs):
            for i in range(len(input_data)):
                input_sample = input_data[i:i+1]  # Use slicing to get a single training sample
                target_sample = target_data[i:i+1]
                neural_network.train(input_sample, target_sample, learning_rate)
   
        neural_network.save_model(MODEL_NAME)

    # Test the trained neural network
    for i in range(len(input_data)):
        input_sample = input_data[i]
        output = neural_network.feedforward(input_sample)
        print(f"Input: {input_sample}, Predicted Output: {output}")
