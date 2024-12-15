import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Generate synthetic binary classification data
# Here we generate random data for demonstration
X = torch.randn(1000, 20)  # 1000 samples, each with 20 features
y = (torch.rand(1000) > 0.5).float()  # 1000 labels (0 or 1)

# 2. Define a simple feedforward neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation
        x = self.fc2(x)  # Output layer
        x = self.sigmoid(x)  # Sigmoid for binary classification
        return x

# 3. Hyperparameters
input_size = 20  # Number of features
hidden_size = 64  # Size of the hidden layer
output_size = 1  # Single output for binary classification
batch_size = 32
epochs = 20
learning_rate = 0.001

# 4. Create a DataLoader for batching
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5. Instantiate the model, loss function, and optimizer
model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. Training loop with corrected accuracy calculation
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        # Calculate the loss
        loss = criterion(outputs.squeeze(), labels)  # .squeeze() to remove extra dimension

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track loss
        running_loss += loss.item()

        # Calculate accuracy
        predicted = (outputs.squeeze() > 0.5).float()  # Squeeze to make outputs a 1D tensor
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Print the training statistics
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total  # Ensure the accuracy is between 0 and 100
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

print("Training complete!")
