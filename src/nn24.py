import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Define a simple feed-forward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Weight initialization functions
def initialize_weights_xavier(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

def initialize_weights_he(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)

# Create synthetic data (e.g., for binary classification)
def generate_data(n_samples=1000, input_size=20):
    X = torch.randn(n_samples, input_size)
    Y = (torch.sum(X, dim=1) > 0).long()  # Target is 1 if sum of input > 0, else 0
    return X, Y

# Training function
def train_model(model, criterion, optimizer, train_loader, epochs=20):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_losses.append(running_loss / len(train_loader))
        accuracy = correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}')
    return train_losses

# Hyperparameters
input_size = 20
hidden_size = 50
output_size = 2
batch_size = 64
epochs = 20

# Generate synthetic dataset
X, Y = generate_data(n_samples=1000, input_size=input_size)
dataset = TensorDataset(X, Y)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
model_xavier = SimpleNN(input_size, hidden_size, output_size)
initialize_weights_xavier(model_xavier)

model_he = SimpleNN(input_size, hidden_size, output_size)
initialize_weights_he(model_he)

# Criterion and optimizer
criterion = nn.CrossEntropyLoss()

# Training with Xavier Initialization
optimizer_xavier = optim.Adam(model_xavier.parameters(), lr=0.001)
print("Training with Xavier Initialization:")
train_losses_xavier = train_model(model_xavier, criterion, optimizer_xavier, train_loader, epochs)

# Training with He Initialization
optimizer_he = optim.Adam(model_he.parameters(), lr=0.001)
print("\nTraining with He Initialization:")
train_losses_he = train_model(model_he, criterion, optimizer_he, train_loader, epochs)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(train_losses_xavier, label='Xavier Initialization')
plt.plot(train_losses_he, label='He Initialization')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison: Xavier vs He Initialization')
plt.legend()
plt.show()
