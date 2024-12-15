import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases with small random values
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))
    
    def forward(self, X):
        # Forward propagation
        self.hidden = sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output = sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return self.output
    
    def train(self, X, y, epochs=10000, learning_rate=0.1):
        for _ in range(epochs):
            # Forward propagation
            hidden = sigmoid(np.dot(X, self.weights1) + self.bias1)
            output = sigmoid(np.dot(hidden, self.weights2) + self.bias2)
            
            # Backpropagation
            output_error = y - output
            output_delta = output_error * sigmoid_derivative(output)
            
            hidden_error = np.dot(output_delta, self.weights2.T)
            hidden_delta = hidden_error * sigmoid_derivative(hidden)
            
            # Update weights and biases
            self.weights2 += learning_rate * np.dot(hidden.T, output_delta)
            self.bias2 += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
            
            self.weights1 += learning_rate * np.dot(X.T, hidden_delta)
            self.bias1 += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
        
        return self

# XOR problem dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train the neural network
nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(X, y)

# Visualization of decision boundary
def plot_decision_boundary(X, nn):
    # Create a mesh to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Predict for each point in the mesh
    Z = nn.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    
    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.RdBu, edgecolors='black')
    
    plt.title('XOR Problem - Neural Network Decision Boundary')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.tight_layout()
    plt.show()

# Visualize the decision boundary
plot_decision_boundary(X, nn)

# Print final predictions to verify XOR logic
print("Final Predictions:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Prediction: {nn.forward(X[i].reshape(1, -1))[0][0]:.4f}")