import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 10
dropout_rate = 0.5

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, use_dropout):
        super(MLP, self).__init__()
        self.use_dropout = use_dropout
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc3(x)
        return x

# Training function
def train(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total

# Validation function
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models
model_without_dropout = MLP(use_dropout=False).to(device)
model_with_dropout = MLP(use_dropout=True).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_without_dropout = optim.Adam(model_without_dropout.parameters(), lr=learning_rate)
optimizer_with_dropout = optim.Adam(model_with_dropout.parameters(), lr=learning_rate)

# Training and validation
results = {"without_dropout": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []},
           "with_dropout": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}}

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    # Without dropout
    train_loss, train_acc = train(model_without_dropout, train_loader, optimizer_without_dropout, criterion)
    val_loss, val_acc = validate(model_without_dropout, test_loader, criterion)

    results["without_dropout"]["train_loss"].append(train_loss)
    results["without_dropout"]["val_loss"].append(val_loss)
    results["without_dropout"]["train_acc"].append(train_acc)
    results["without_dropout"]["val_acc"].append(val_acc)

    # With dropout
    train_loss, train_acc = train(model_with_dropout, train_loader, optimizer_with_dropout, criterion)
    val_loss, val_acc = validate(model_with_dropout, test_loader, criterion)

    results["with_dropout"]["train_loss"].append(train_loss)
    results["with_dropout"]["val_loss"].append(val_loss)
    results["with_dropout"]["train_acc"].append(train_acc)
    results["with_dropout"]["val_acc"].append(val_acc)

# Plot results
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(results["without_dropout"]["train_loss"], label="Train Loss (No Dropout)")
plt.plot(results["without_dropout"]["val_loss"], label="Val Loss (No Dropout)")
plt.plot(results["with_dropout"]["train_loss"], label="Train Loss (Dropout)")
plt.plot(results["with_dropout"]["val_loss"], label="Val Loss (Dropout)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss vs. Epoch")

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(results["without_dropout"]["train_acc"], label="Train Acc (No Dropout)")
plt.plot(results["without_dropout"]["val_acc"], label="Val Acc (No Dropout)")
plt.plot(results["with_dropout"]["train_acc"], label="Train Acc (Dropout)")
plt.plot(results["with_dropout"]["val_acc"], label="Val Acc (Dropout)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy vs. Epoch")

plt.tight_layout()
plt.show()
