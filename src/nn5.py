import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class CircleClassificationMLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with small random values
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.1
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.1
        self.bias2 = np.zeros((1, output_size))
    
    def forward(self, X):
        # Forward propagation
        self.hidden_layer = sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output_layer = sigmoid(np.dot(self.hidden_layer, self.weights2) + self.bias2)
        return self.output_layer
    
    def train(self, X, y, epochs=10000, learning_rate=0.1):
        for _ in range(epochs):
            # Forward propagation
            hidden_layer = sigmoid(np.dot(X, self.weights1) + self.bias1)
            output_layer = sigmoid(np.dot(hidden_layer, self.weights2) + self.bias2)
            
            # Backpropagation
            output_error = y - output_layer
            output_delta = output_error * sigmoid_derivative(output_layer)
            
            hidden_error = np.dot(output_delta, self.weights2.T)
            hidden_delta = hidden_error * sigmoid_derivative(hidden_layer)
            
            # Update weights and biases
            self.weights2 += learning_rate * np.dot(hidden_layer.T, output_delta)
            self.bias2 += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
            
            self.weights1 += learning_rate * np.dot(X.T, hidden_delta)
            self.bias1 += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
        
        return self

# Generate dataset: points inside/outside a circle
def generate_circle_dataset(num_samples=500, radius=1):
    # Generate random points in a 2x2 square
    X = np.random.uniform(-1.5, 1.5, (num_samples, 2))
    
    # Label points based on whether they're inside the unit circle
    y = (X[:, 0]**2 + X[:, 1]**2 <= radius**2).astype(float).reshape(-1, 1)
    
    return X, y

# Generate the dataset
X, y = generate_circle_dataset()

# Create and train the neural network
np.random.seed(42)  # For reproducibility
mlp = CircleClassificationMLP(input_size=2, hidden_size=8, output_size=1)
mlp.train(X, y)

# Visualization function
def plot_decision_boundary(X, y, mlp):
    # Create a mesh to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Predict for each point in the mesh
    Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    
    # Plot the training points
    plt.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1], 
                c='blue', label='Inside Circle', edgecolors='black')
    plt.scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1], 
                c='red', label='Outside Circle', edgecolors='black')
    
    plt.title('MLP Circle Classification')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Visualize the decision boundary
plot_decision_boundary(X, y, mlp)

# Compute and print accuracy
predictions = mlp.forward(X)
accuracy = np.mean((predictions >= 0.5) == y)
print(f"Training Accuracy: {accuracy * 100:.2f}%")