import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),  # Flatten the 28x28 image into a 1D vector
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),  # Output size is same as the input size
            nn.Sigmoid()  # Output pixels in the range [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Prepare the Data (Using MNIST dataset)
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Initialize the model, loss function, and optimizer
model = Autoencoder()
criterion = nn.MSELoss()  # Mean Squared Error loss for reconstruction
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for data in train_loader:
        images, _ = data  # We only need the images, not the labels
        images = images.view(images.size(0), -1)  # Flatten images to 1D vectors of size 28*28

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, images)  # Compute the reconstruction loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Visualize the reconstructed images
model.eval()  # Set the model to evaluation mode

# Get a batch of test images
data_iter = iter(train_loader)
images, _ = next(data_iter)

# Flatten the images
images_flat = images.view(images.size(0), -1)

# Get the reconstructed images
reconstructed = model(images_flat)

# Plot original and reconstructed images side by side
n = 10  # Number of images to display
plt.figure(figsize=(10, 4))
for i in range(n):
    # Original image
    ax = plt.subplot(2, n, i+1)
    plt.imshow(images[i].numpy().reshape(28, 28), cmap='gray')
    ax.axis('off')

    # Reconstructed image
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(reconstructed[i].detach().numpy().reshape(28, 28), cmap='gray')
    ax.axis('off')

plt.show()

