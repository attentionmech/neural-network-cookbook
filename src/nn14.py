import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# 1. Generate synthetic data for multi-class classification (3 classes)
num_samples = 1000
num_features = 20
num_classes = 3

# Random features
X = torch.randn(num_samples, num_features)

# Random labels for 3 classes (0, 1, or 2)
y = torch.randint(0, num_classes, (num_samples,))

# 2. Split the data into training and validation sets (80/20 split)
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 3. Define the neural network model
class MultiClassNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiClassNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)  # Output layer should have 'num_classes' units
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation function for hidden layer
        x = self.fc2(x)  # Output layer (logits for each class)
        return x  # We will apply softmax during loss calculation

# 4. Hyperparameters
input_size = num_features
hidden_size = 64
output_size = num_classes
batch_size = 32
epochs = 200
learning_rate = 0.001

# 5. Instantiate the model, loss function, and optimizer
model = MultiClassNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()  # This loss function combines softmax and cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. Function to evaluate the model on validation data
def evaluate_model(model, val_loader):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No need to compute gradients for evaluation
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)  # Get the class with the highest score
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_val_loss, accuracy

# 7. Training loop
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
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track loss
        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest score
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Calculate training loss and accuracy
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    
    # Evaluate the model on the validation set
    val_loss, val_accuracy = evaluate_model(model, val_loader)
    
    # Print statistics
    print(f"Epoch [{epoch+1}/{epochs}], "
          f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

print("Training complete!")
