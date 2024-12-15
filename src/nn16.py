import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Generate a simple 2D dataset (e.g., points on a 2D plane)
np.random.seed(0)
X = np.random.rand(100, 2)  # 100 samples, 2 features (2D data)

# Step 2: Convert the data into PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)

# Step 3: Create DataLoader for training
train_data = TensorDataset(X_tensor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Step 4: Define the Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder: Reduce the input to a lower-dimensional space (2 -> 1)
        self.encoder = nn.Sequential(
            nn.Linear(2, 1),  # Input size 2 -> Hidden size 1
            nn.ReLU()
        )
        
        # Decoder: Reconstruct the input back to the original dimensions (1 -> 2)
        self.decoder = nn.Sequential(
            nn.Linear(1, 2),  # Hidden size 1 -> Output size 2
            nn.Sigmoid()  # Apply Sigmoid for [0, 1] output range
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Step 5: Initialize the model, loss function, and optimizer
model = Autoencoder()
criterion = nn.MSELoss()  # Mean Squared Error for reconstruction loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Training the Autoencoder
num_epochs = 1000
for epoch in range(num_epochs):
    running_loss = 0.0
    for data in train_loader:
        inputs = data[0]  # Get the input data
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, inputs)  # Calculate reconstruction loss
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update the model's weights
        
        running_loss += loss.item()  # Accumulate loss

    # Print the loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

# Step 7: Visualize the original and reconstructed data
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    reconstructed_data = model(X_tensor).numpy()

# Plot original and reconstructed data
plt.figure(figsize=(10, 5))

# Original data
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], color='blue', label='Original Data')
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Reconstructed data
plt.subplot(1, 2, 2)
plt.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], color='red', label='Reconstructed Data')
plt.title("Reconstructed Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()

