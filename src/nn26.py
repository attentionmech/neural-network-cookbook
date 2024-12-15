import torch
import torch.nn as nn
import torch.optim as optim

# Define the Custom Loss Function (Mean Squared Logarithmic Error - MSLE)
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Compute MSLE (Mean Squared Logarithmic Error)
        log_pred = torch.log1p(y_pred)  # log(1 + y_pred)
        log_true = torch.log1p(y_true)  # log(1 + y_true)
        loss = torch.mean((log_pred - log_true) ** 2)  # Mean Squared Error of log-transformed values
        return loss

# Define the Simple Neural Network for Regression
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # First fully connected layer
        self.fc2 = nn.Linear(64, 64)  # Second fully connected layer
        self.fc3 = nn.Linear(64, 1)  # Output layer
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply ReLU to the first layer
        x = self.relu(self.fc2(x))  # Apply ReLU to the second layer
        x = self.fc3(x)  # Output layer (no activation function)
        return x

# Generate some synthetic data for training
torch.manual_seed(0)  # Set seed for reproducibility
X = torch.linspace(1, 10, 100).view(-1, 1)  # 100 data points from 1 to 10
y = 2 * X + 3 + torch.randn_like(X) * 0.5  # y = 2X + 3 with added Gaussian noise

# Initialize the model, loss function, and optimizer
model = SimpleNN()  # Create an instance of the neural network
criterion = CustomLoss()  # Instantiate the custom loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Training loop
num_epochs = 1000  # Number of epochs to train the model

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    optimizer.zero_grad()  # Zero the gradients before the backward pass

    # Forward pass: Get model predictions
    y_pred = model(X)

    # Compute the loss using the custom loss function
    loss = criterion(y_pred, y)

    # Backward pass: Compute gradients
    loss.backward()

    # Update the model parameters using the optimizer
    optimizer.step()

    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        
# After training, you can use the model to make predictions
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_pred = model(X)

# Example of the predicted vs actual values after training
print("\nPredicted vs Actual:")
for i in range(10):  # Show the first 10 predictions
    print(f"Predicted: {y_pred[i].item():.2f}, Actual: {y[i].item():.2f}")
