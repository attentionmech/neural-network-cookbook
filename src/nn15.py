import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# Generate synthetic data (e.g., 100 samples, 5 features)
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = np.random.randint(0, 2, 100)  # Binary target labels

# Normalize the data using MinMaxScaler (scaling features to [0, 1])
scaler_minmax = MinMaxScaler()
X_minmax_scaled = scaler_minmax.fit_transform(X)

# Alternatively, normalize the data using StandardScaler (zero mean, unit variance)
scaler_standard = StandardScaler()
X_standard_scaled = scaler_standard.fit_transform(X)

# Convert the scaled data into PyTorch tensors
X_minmax_tensor = torch.tensor(X_minmax_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

X_standard_tensor = torch.tensor(X_standard_scaled, dtype=torch.float32)

# Create DataLoader for training
train_data_minmax = TensorDataset(X_minmax_tensor, y_tensor)
train_loader_minmax = DataLoader(train_data_minmax, batch_size=32, shuffle=True)

train_data_standard = TensorDataset(X_standard_tensor, y_tensor)
train_loader_standard = DataLoader(train_data_standard, batch_size=32, shuffle=True)

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(5, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 2)  # Binary classification
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Train the model using MinMax normalized data
print("Training with MinMax Scaled Data:")
train(model, train_loader_minmax, criterion, optimizer)

# Train the model using StandardScaler normalized data
print("Training with Standard Scaled Data:")
train(model, train_loader_standard, criterion, optimizer)

