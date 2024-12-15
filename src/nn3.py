import math
import random

# Step 1: Define the sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Step 2: Initialize the MLP structure
# XOR input data
X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
y = [0, 1, 1, 0]  # XOR outputs

# Hyperparameters
learning_rate = 0.1  # Reduced learning rate
epochs = 20000  # Increased number of epochs

# Step 3: Randomly initialize weights and biases for the MLP
# Input to Hidden layer weights (2 input neurons -> 2 hidden neurons)
weights_input_hidden = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]  # 2x2 matrix
bias_hidden = [random.uniform(-1, 1) for _ in range(2)]  # Bias for each hidden neuron

# Hidden to Output layer weights (2 hidden neurons -> 1 output neuron)
weights_hidden_output = [random.uniform(-1, 1) for _ in range(2)]  # 2x1 vector
bias_output = random.uniform(-1, 1)  # Bias for output neuron

# Step 4: Train the MLP
for epoch in range(epochs):
    total_error = 0  # To accumulate the total error for the current epoch

    for i in range(len(X)):
        # Forward pass:
        # Input to Hidden Layer
        input_layer = X[i]
        hidden_layer_input = [sum(input_layer[j] * weights_input_hidden[j][k] for j in range(2)) + bias_hidden[k] for k in range(2)]
        hidden_layer_output = [sigmoid(h) for h in hidden_layer_input]

        # Hidden to Output Layer
        output_layer_input = sum(hidden_layer_output[k] * weights_hidden_output[k] for k in range(2)) + bias_output
        output = sigmoid(output_layer_input)

        # Calculate the error (output - expected)
        error = y[i] - output
        total_error += error ** 2  # Sum of squared errors for the epoch

        # Backpropagation:
        # Calculate the gradient for output layer
        output_delta = error * sigmoid_derivative(output)

        # Calculate the gradient for hidden layer
        hidden_deltas = [output_delta * weights_hidden_output[k] * sigmoid_derivative(hidden_layer_output[k]) for k in range(2)]

        # Update weights and biases (Gradient Descent)
        for j in range(2):
            weights_hidden_output[j] += learning_rate * output_delta * hidden_layer_output[j]
        bias_output += learning_rate * output_delta

        for k in range(2):
            for j in range(2):
                weights_input_hidden[j][k] += learning_rate * hidden_deltas[k] * input_layer[j]
        for k in range(2):
            bias_hidden[k] += learning_rate * hidden_deltas[k]

    # Calculate the mean squared error for this epoch
    mean_squared_error = total_error / len(X)
    
    # Print the error progress every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}/{epochs} - MSE: {mean_squared_error:.5f}")

# Step 5: Test the trained MLP
print("\nTrained weights (input to hidden):", weights_input_hidden)
print("Trained bias (hidden):", bias_hidden)
print("Trained weights (hidden to output):", weights_hidden_output)
print("Trained bias (output):", bias_output)

# Test the model on all XOR inputs
for i in range(len(X)):
    input_layer = X[i]
    hidden_layer_input = [sum(input_layer[j] * weights_input_hidden[j][k] for j in range(2)) + bias_hidden[k] for k in range(2)]
    hidden_layer_output = [sigmoid(h) for h in hidden_layer_input]
    
    output_layer_input = sum(hidden_layer_output[k] * weights_hidden_output[k] for k in range(2)) + bias_output
    output = sigmoid(output_layer_input)
    
    # Apply threshold (0.5) to classify the output as either 0 or 1
    output = 1 if output >= 0.5 else 0
    
    print(f"Input: {X[i]} -> Predicted Output: {output} (Expected: {y[i]})")
