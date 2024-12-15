import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate a simple dataset: a sine wave
seq_length = 10
num_samples = 1000
def create_dataset():
    x = np.linspace(0, 100, num_samples)
    data = np.sin(x)
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

# Dataset preparation
sequences, targets = create_dataset()
train_data = torch.tensor(sequences, dtype=torch.float32)
train_labels = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

# Hyperparameters
input_size = 1
hidden_size = 50
output_size = 1
num_layers = 1
learning_rate = 0.001
epochs = 50
batch_size = 64

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # Initialize hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # Initialize cell state
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the last output for prediction
        return out

# Initialize the model, loss function, and optimizer
model = LSTM(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Data loader
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.view(-1, seq_length, input_size)  # Reshape for LSTM
        targets = targets

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Testing the model
model.eval()
test_seq = torch.tensor(sequences[:1], dtype=torch.float32).view(-1, seq_length, input_size)
with torch.no_grad():
    prediction = model(test_seq).item()
print(f"Next value prediction: {prediction:.4f}")
