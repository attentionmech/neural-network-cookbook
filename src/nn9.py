# Re-import necessary libraries after environment reset
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define a simple MLP model in PyTorch with sigmoid activation
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.hidden(x))  # Apply sigmoid activation in the hidden layer
        x = self.output(x)
        return x

# Initialize model, loss function, and optimizer
torch.manual_seed(42)
input_size = 1
hidden_size = 10
output_size = 1
model = SimpleMLP(input_size, hidden_size, output_size)

# Generate synthetic data
x = torch.linspace(-10, 10, 200).view(-1, 1)  # Input values
y = 1 / (1 + torch.exp(-x))  # Sigmoid function applied to inputs

# Forward pass through the model
with torch.no_grad():  # Disable gradient calculation for visualization
    hidden_activations = model.sigmoid(model.hidden(x)).numpy()  # Hidden layer activations
    predictions = model(x).numpy()  # Final output

# Visualize sigmoid activation and model predictions
plt.figure(figsize=(12, 6))

# Original sigmoid function
plt.subplot(1, 2, 1)
plt.plot(x.numpy(), y.numpy(), label="True Sigmoid", color="blue")
plt.title("True Sigmoid Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()

# Sigmoid activations of the hidden layer neurons
plt.subplot(1, 2, 2)
for i in range(hidden_size):
    plt.plot(x.numpy(), hidden_activations[:, i], label=f"Neuron {i+1}", alpha=0.6)
plt.title("Sigmoid Activations in Hidden Layer")
plt.xlabel("Input")
plt.ylabel("Activation")
plt.legend()

plt.tight_layout()
plt.show()
