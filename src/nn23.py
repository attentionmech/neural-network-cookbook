import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the generator network
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, z):
        return self.net(z)

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Define real data distribution
def real_data_distribution(batch_size):
    # Generate points in a 2D circular distribution
    theta = 2 * np.pi * np.random.rand(batch_size)
    r = np.random.normal(loc=5.0, scale=0.5, size=batch_size)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)

# Training settings
batch_size = 128
z_dim = 2
lr = 0.0002
epochs = 15000

# Initialize networks and optimizers
generator = Generator(input_dim=z_dim, output_dim=2)
discriminator = Discriminator(input_dim=2)

optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

criterion = nn.BCELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

# Training loop
for epoch in range(epochs):
    # Train Discriminator
    real_data = torch.tensor(real_data_distribution(batch_size), dtype=torch.float32).to(device)
    fake_data = generator(torch.randn(batch_size, z_dim).to(device)).detach()

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    optimizer_D.zero_grad()
    
    real_loss = criterion(discriminator(real_data), real_labels)
    fake_loss = criterion(discriminator(fake_data), fake_labels)
    
    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    z = torch.randn(batch_size, z_dim).to(device)
    fake_data = generator(z)
    fake_labels = torch.ones(batch_size, 1).to(device)

    optimizer_G.zero_grad()
    g_loss = criterion(discriminator(fake_data), fake_labels)
    g_loss.backward()
    optimizer_G.step()

    # Logging
    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}]  D Loss: {d_loss.item():.4f}  G Loss: {g_loss.item():.4f}")

# Visualization of results
z = torch.randn(1000, z_dim).to(device)
with torch.no_grad():
    generated_data = generator(z).cpu().numpy()

real_data_samples = real_data_distribution(1000)

plt.figure(figsize=(8, 8))
plt.scatter(real_data_samples[:, 0], real_data_samples[:, 1], color='blue', alpha=0.5, label='Real Data')
plt.scatter(generated_data[:, 0], generated_data[:, 1], color='red', alpha=0.5, label='Generated Data')
plt.legend()
plt.title("Real vs Generated Data")
plt.show()
