import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse_loss(y_true, y_pred):
    return 0.5 * (y_true - y_pred)**2

def mse_loss_derivative(y_true, y_pred):
    return -(y_true - y_pred)

class MLP:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        # Initialize weights and biases with random values
        self.weights1 = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden1_size)]
        self.biases1 = [random.uniform(-1, 1) for _ in range(hidden1_size)]
        
        self.weights2 = [[random.uniform(-1, 1) for _ in range(hidden1_size)] for _ in range(hidden2_size)]
        self.biases2 = [random.uniform(-1, 1) for _ in range(hidden2_size)]

        self.weights3 = [[random.uniform(-1, 1) for _ in range(hidden2_size)] for _ in range(output_size)]
        self.biases3 = [random.uniform(-1, 1) for _ in range(output_size)]


    def forward(self, input_data):
        # Hidden Layer 1
        self.hidden1_inputs = [sum(self.weights1[i][j] * input_data[j] for j in range(len(input_data))) + self.biases1[i]
                                for i in range(len(self.weights1))]
        self.hidden1_outputs = [sigmoid(x) for x in self.hidden1_inputs]

        # Hidden Layer 2
        self.hidden2_inputs = [sum(self.weights2[i][j] * self.hidden1_outputs[j] for j in range(len(self.hidden1_outputs))) + self.biases2[i]
                                for i in range(len(self.weights2))]
        self.hidden2_outputs = [sigmoid(x) for x in self.hidden2_inputs]

        # Output Layer
        self.output_inputs = [sum(self.weights3[i][j] * self.hidden2_outputs[j] for j in range(len(self.hidden2_outputs))) + self.biases3[i]
                                for i in range(len(self.weights3))]
        self.output = [sigmoid(x) for x in self.output_inputs]
        
        return self.output

    def backward(self, input_data, target_data, learning_rate):
        # Output Layer Backpropagation
        output_error = [mse_loss_derivative(target_data[i], self.output[i]) * sigmoid_derivative(self.output_inputs[i])
                        for i in range(len(target_data))]
        
        # Update the weights and biases
        for i in range(len(self.weights3)):
            for j in range(len(self.weights3[0])):
                self.weights3[i][j] -= learning_rate * output_error[i] * self.hidden2_outputs[j]
            self.biases3[i] -= learning_rate * output_error[i]
        
        # Hidden Layer 2 Backpropagation
        hidden2_errors = [sum(output_error[k] * self.weights3[k][i] for k in range(len(output_error))) * sigmoid_derivative(self.hidden2_inputs[i]) 
                          for i in range(len(self.hidden2_outputs))]

        for i in range(len(self.weights2)):
            for j in range(len(self.weights2[0])):
                 self.weights2[i][j] -= learning_rate * hidden2_errors[i] * self.hidden1_outputs[j]
            self.biases2[i] -= learning_rate * hidden2_errors[i]

        # Hidden Layer 1 Backpropagation
        hidden1_errors = [sum(hidden2_errors[k] * self.weights2[k][i] for k in range(len(hidden2_errors))) * sigmoid_derivative(self.hidden1_inputs[i])
                          for i in range(len(self.hidden1_outputs))]

        for i in range(len(self.weights1)):
            for j in range(len(self.weights1[0])):
                self.weights1[i][j] -= learning_rate * hidden1_errors[i] * input_data[j]
            self.biases1[i] -= learning_rate * hidden1_errors[i]
    
    def train(self, training_data, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            for input_data, target_data in training_data:
                 output = self.forward(input_data)
                 loss = sum([mse_loss(target_data[i], output[i]) for i in range(len(target_data))])
                 total_loss += loss
                 self.backward(input_data, target_data, learning_rate)

            if epoch % 1000 == 0:
                print(f"Epoch: {epoch}, Loss: {total_loss / len(training_data)}")



# XOR Training Data
training_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]

# Initialize the MLP
input_size = 2
hidden1_size = 4
hidden2_size = 4
output_size = 1
mlp = MLP(input_size, hidden1_size, hidden2_size, output_size)

# Training
epochs = 10000
learning_rate = 0.1
mlp.train(training_data, epochs, learning_rate)

# Test
print("\nTesting:")
for input_data, target_data in training_data:
    output = mlp.forward(input_data)
    print(f"Input: {input_data}, Predicted: {output[0]:.4f}, Target: {target_data[0]}")