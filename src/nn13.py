import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# 1. Generate synthetic binary classification data
X = torch.randn(1000, 20)  # 1000 samples, each with 20 features
y = (torch.rand(1000) > 0.5).float()  # 1000 binary labels (0 or 1)

# 2. Split the data into training and validation sets (80/20 split)
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 3. Define a simple feedforward neural network model
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

# 4. Hyperparameters
input_size = 20
hidden_size = 64
output_size = 1
batch_size = 32
epochs = 50  # Max epochs before stopping early
learning_rate = 0.001
patience = 5  # How many epochs to wait for improvement

# 5. Instantiate the model, loss function, and optimizer
model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification
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
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()
            
            predicted = (outputs.squeeze() > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_val_loss, accuracy

# 7. Training loop with early stopping
best_val_loss = float('inf')  # Initialize with a very high value
epochs_without_improvement = 0  # Counter to track the number of epochs without improvement

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
        loss = criterion(outputs.squeeze(), labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track loss
        running_loss += loss.item()

        # Calculate accuracy
        predicted = (outputs.squeeze() > 0.5).float()  # Convert to binary predictions
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
    
    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0  # Reset the counter
        # Optionally, save the best model's weights
        torch.save(model.state_dict(), '/tmp/best_model.pth')
        print("Validation loss improved, saving model!")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered. Validation loss didn't improve for {patience} epochs.")
            break  # Stop training

print("Training complete!")
