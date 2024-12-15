import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import KFold
import numpy as np

# Example dataset class (you can replace this with your custom dataset)
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Define a simple feed-forward neural network for demonstration
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

# Function to train the model
def train_model(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        # Assuming inputs are of shape (batch_size, input_dim)
        # Move to the device (GPU/CPU)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

# Function to evaluate the model
def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example dataset (you can replace this with your dataset)
# Let's assume each sample is a vector of length 10, and we have 1000 samples.
data = np.random.rand(1000, 10).astype(np.float32)
labels = np.random.randint(0, 2, size=(1000,))

# Create dataset
dataset = CustomDataset(torch.tensor(data), torch.tensor(labels))

# Parameters for the model
input_dim = 10  # Input feature dimension
hidden_dim = 64  # Hidden layer size
output_dim = 2  # Number of classes (binary classification)

# Number of folds for cross-validation
k_folds = 5

# Set up k-fold cross-validation
kf = KFold(n_splits=k_folds, shuffle=True)

# Store the results of each fold
fold_results = []

# k-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"Fold {fold+1}/{k_folds}")

    # Create data loaders for training and validation
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
    
    train_loader = DataLoader(dataset, batch_size=32, sampler=train_subsampler)
    val_loader = DataLoader(dataset, batch_size=32, sampler=val_subsampler)

    # Initialize the model
    model = SimpleNN(input_dim, hidden_dim, output_dim).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model for some epochs
    epochs = 10
    for epoch in range(epochs):
        train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save the results of this fold
    fold_results.append({
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

# After all folds, calculate the average performance across all folds
avg_train_loss = np.mean([result["train_loss"] for result in fold_results])
avg_train_accuracy = np.mean([result["train_accuracy"] for result in fold_results])
avg_val_loss = np.mean([result["val_loss"] for result in fold_results])
avg_val_accuracy = np.mean([result["val_accuracy"] for result in fold_results])

print("\nAverage Performance Across All Folds:")
print(f"Avg Train Loss: {avg_train_loss:.4f}, Avg Train Accuracy: {avg_train_accuracy:.4f}")
print(f"Avg Val Loss: {avg_val_loss:.4f}, Avg Val Accuracy: {avg_val_accuracy:.4f}")
