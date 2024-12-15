import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network for function approximation
class FunctionApproximator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FunctionApproximator, self).__init__()
        # Define layers of the neural network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)  # Using ReLU activation for hidden layer 1
        x = self.fc2(x)
        x = self.relu(x)  # Using ReLU activation for hidden layer 2
        x = self.fc3(x)
        return x  # No activation in the output layer for regression

# Generate data for approximating sin(x)
def generate_data(start=-10, end=10, num_samples=1000):
    x = np.linspace(start, end, num_samples)
    y = np.sin(x)
    return torch.tensor(x, dtype=torch.float32).view(-1, 1), torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Hyperparameters
input_dim = 1          # Input is a single scalar (x)
hidden_dim = 64        # Hidden layer size
output_dim = 1         # Output is a single scalar (y = sin(x))
learning_rate = 0.001  # Learning rate
epochs = 50000          # Number of training epochs

# Generate training data
x_train, y_train = generate_data()

# Define the model, loss function, and optimizer
model = FunctionApproximator(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()

    # Forward pass
    y_pred = model(x_train)

    # Compute loss
    loss = criterion(y_pred, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Compute new gradients
    optimizer.step()       # Update the weights

    # Print loss every 500 epochs
    if epoch % 500 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_pred = model(x_train)

# Convert predictions and targets to numpy for plotting
x_train = x_train.numpy()
y_train = y_train.numpy()
y_pred = y_pred.numpy()

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train, label="True sin(x)", color="blue")
plt.plot(x_train, y_pred, label="Predicted sin(x)", color="red", linestyle="--")
plt.title("Function Approximation: Neural Network vs True sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
