import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define a simple feedforward neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for batching
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define a function to train a model
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Define a function to evaluate a model
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.numpy())
            all_labels.append(labels.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# Initialize models
input_size = X_train.shape[1]
hidden_size = 32
output_size = 2

# Create a list of models (let's train 3 models)
models = [SimpleNN(input_size, hidden_size, output_size) for _ in range(3)]

# Create optimizers and loss functions for each model
optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]
criterion = nn.CrossEntropyLoss()

# Train individual models
for i, model in enumerate(models):
    print(f"Training model {i + 1}")
    train_model(model, train_loader, criterion, optimizers[i], epochs=10)

# Evaluate individual models
individual_accuracies = []
for i, model in enumerate(models):
    accuracy = evaluate_model(model, test_loader)
    individual_accuracies.append(accuracy)
    print(f"Model {i + 1} accuracy: {accuracy:.4f}")

# Ensemble prediction (majority voting)
def ensemble_predict(models, inputs):
    outputs = [model(inputs) for model in models]
    outputs = torch.stack(outputs, dim=0)
    outputs = torch.mean(outputs, dim=0)
    _, predicted = torch.max(outputs, 1)
    return predicted

# Evaluate ensemble model
def evaluate_ensemble(models, test_loader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            preds = ensemble_predict(models, inputs)
            all_preds.append(preds.numpy())
            all_labels.append(labels.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

ensemble_accuracy = evaluate_ensemble(models, test_loader)
print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")

# Compare individual model accuracies with ensemble accuracy
for i, accuracy in enumerate(individual_accuracies):
    print(f"Model {i + 1} accuracy: {accuracy:.4f}")
print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
