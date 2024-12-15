import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

# Define a simple model
model = nn.Sequential(
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Define a StepLR scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Dummy dataset
x_train = torch.rand((1000, 28*28))
y_train = torch.randint(0, 10, (1000,))

# Training loop
for epoch in range(20):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Step the scheduler
    scheduler.step()

    # Print learning rate
    print(f"Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()}")
