import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Create a dataset class to generate random images
class RandomImageDataset(Dataset):
    def __init__(self, num_samples, image_size, num_classes, transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create a random image (3 channels, height x width)
        image = np.random.rand(3, self.image_size, self.image_size) * 255
        image = Image.fromarray(image.astype('uint8').transpose(1, 2, 0))  # Convert to RGB image
        
        # Random class label
        label = np.random.randint(0, self.num_classes)

        if self.transform:
            image = self.transform(image)

        return image, label

# Set parameters
image_size = 224  # Image size expected by most pre-trained models
num_classes = 2  # Binary classification
num_train_samples = 1000
num_val_samples = 200

# Define the transforms for image preprocessing
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create random datasets
train_dataset = RandomImageDataset(num_samples=num_train_samples, image_size=image_size, num_classes=num_classes, transform=transform)
val_dataset = RandomImageDataset(num_samples=num_val_samples, image_size=image_size, num_classes=num_classes, transform=transform)

# Create DataLoaders for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Freeze all layers except the final fully connected layer
for param in model.parameters():
    param.requires_grad = False

# Modify the final fully connected layer to match the number of classes in your dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Clear gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        running_loss += loss.item()
        
        # Track accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Evaluation loop
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # Disable gradient calculation for evaluation
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_accuracy = 100 * correct / total
print(f'Validation Accuracy: {val_accuracy:.2f}%')

# Save the fine-tuned model
torch.save(model.state_dict(), 'fine_tuned_resnet18.pth')

# Display a random image from the dataset
random_img, _ = train_dataset[0]
plt.imshow(random_img.permute(1, 2, 0))
plt.title("Random Image from Training Set")
plt.show()
