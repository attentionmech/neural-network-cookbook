import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# Basic CNN model without Batch Normalization
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Instantiate the model, loss, and optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")

# Train the model without Batch Normalization
print("Training without Batch Normalization:")
train_model(model, train_loader, criterion, optimizer, epochs=5)


print("\n\n-----\n\n")

# CNN model with Batch Normalization
class SimpleNetWithBN(nn.Module):
    def __init__(self):
        super(SimpleNetWithBN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.bn1 = nn.BatchNorm1d(128)  # Batch Normalization layer
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)  # Batch Normalization layer
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = torch.relu(self.bn1(self.fc1(x)))  # Apply BN after the first layer
        x = torch.relu(self.bn2(self.fc2(x)))  # Apply BN after the second layer
        x = self.fc3(x)
        return x

# Instantiate the model, loss, and optimizer with Batch Normalization
model_with_bn = SimpleNetWithBN()
optimizer_with_bn = optim.SGD(model_with_bn.parameters(), lr=0.01, momentum=0.9)

# Train the model with Batch Normalization
print("Training with Batch Normalization:")
train_model(model_with_bn, train_loader, criterion, optimizer_with_bn, epochs=5)

