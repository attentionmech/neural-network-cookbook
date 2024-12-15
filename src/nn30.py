import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import numpy as np
import torch
from PIL import Image, ImageDraw
import random
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# 1. Define the Siamese Network Architecture

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Shared CNN model
        self.conv1 = nn.Conv2d(1, 64, kernel_size=10)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4)
        self.fc1 = nn.Linear(128 * 3 * 3, 1024)  # Flatten the output and create a fully connected layer
        self.fc2 = nn.Linear(1024, 256)
        
    def forward_once(self, x):
        # Forward pass for one input (same for both inputs)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)  # Flatten the output of the convolutional layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    
    def forward(self, input1, input2):
        # Forward pass for both inputs
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# 2. Contrastive Loss Function

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        # Euclidean distance
        euclidean_distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
        
        # Contrastive loss function
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# 3. Dataset and DataLoader

class SiameseDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img1, img2, label = self.data[idx]
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label

# 4. Training Loop

def train_siamese_network(model, train_loader, optimizer, loss_fn, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (img1, img2, labels) in enumerate(train_loader):
            img1, img2, labels = img1.cuda(), img2.cuda(), labels.cuda()

            optimizer.zero_grad()

            # Forward pass
            output1, output2 = model(img1, img2)

            # Compute loss
            loss = loss_fn(output1, output2, labels)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader)}')

# 5. Example Usage

# Example data format: [(image1, image2, label), ...]
# 'label' is 1 if the images are similar, 0 if they are dissimilar.
# Here, img1 and img2 should be PIL image objects (or similar format).

# Create a dummy dataset (replace with actual image data)

# Function to generate a random image with a circle or a square
def generate_image(shape_type="circle"):
    img = Image.new('L', (100, 100), color=255)  # Create a white image of size 100x100
    draw = ImageDraw.Draw(img)
    
    if shape_type == "circle":
        radius = random.randint(10, 30)
        x = random.randint(radius, 100 - radius)
        y = random.randint(radius, 100 - radius)
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=random.randint(0, 255))
    
    elif shape_type == "square":
        side = random.randint(20, 50)
        x = random.randint(0, 100 - side)
        y = random.randint(0, 100 - side)
        draw.rectangle([x, y, x + side, y + side], fill=random.randint(0, 255))
    
    return img

# Function to generate a pair of images and a label
def generate_image_pair():
    # Randomly choose the type of shapes for the pair
    shape1 = random.choice(["circle", "square"])
    shape2 = random.choice(["circle", "square"])

    # Generate images
    img1 = generate_image(shape1)
    img2 = generate_image(shape2)

    # Label 1 if similar (both are the same shape), 0 if dissimilar (different shapes)
    label = 1 if shape1 == shape2 else 0

    return img1, img2, label

# Generate a batch of image pairs
def generate_image_batch(batch_size):
    data = []
    for _ in range(batch_size):
        img1, img2, label = generate_image_pair()
        data.append((img1, img2, label))
    return data

# Visualize a few image pairs
def visualize_pairs(batch_data):
    fig, axes = plt.subplots(len(batch_data), 2, figsize=(5, 5*len(batch_data)))
    for i, (img1, img2, label) in enumerate(batch_data):
        axes[i][0].imshow(img1, cmap='gray')
        axes[i][0].axis('off')
        axes[i][1].imshow(img2, cmap='gray')
        axes[i][1].axis('off')
        axes[i][0].set_title(f"Image 1")
        axes[i][1].set_title(f"Image 2\nLabel: {label}")
    plt.show()

# Generate a batch of synthetic pairs
batch_size = 5
batch_data = generate_image_batch(batch_size)
# visualize_pairs(batch_data)

# Optionally, you can transform them into tensors for training
transform = transforms.Compose([
    transforms.ToTensor()
])

# Apply the transformation and print some example pairs
train_data = [(transform(img1), transform(img2), label) for img1, img2, label in batch_data]

# Check a transformed example
print(train_data[0])  # Show the first example in transformed tensor form

# Transformation to apply to images
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

train_dataset = SiameseDataset(train_data, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
model = SiameseNetwork().cuda()
loss_fn = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the network
train_siamese_network(model, train_loader, optimizer, loss_fn, num_epochs=10)
