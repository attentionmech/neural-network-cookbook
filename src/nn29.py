import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
import string



class SimpleAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(SimpleAttention, self).__init__()
        # Define the linear layers to compute Q, K, V
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)

    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)

        # Compute Q, K, V
        Q = self.query(x)  # (batch_size, seq_length, attention_dim)
        K = self.key(x)    # (batch_size, seq_length, attention_dim)
        V = self.value(x)  # (batch_size, seq_length, attention_dim)

        # Scaled dot-product attention
        scores = torch.bmm(Q, K.transpose(1, 2))  # (batch_size, seq_length, seq_length)
        # Scale the scores by the square root of the attention dimension
        scores = scores / torch.sqrt(torch.tensor(K.size(-1), dtype=torch.float32))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_length, seq_length)

        # Compute the context vector as the weighted sum of values
        output = torch.bmm(attention_weights, V)  # (batch_size, seq_length, attention_dim)

        return output, attention_weights




# Step 1: Dummy dataset generation (random sequences and labels)
class RandomTextDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=10, vocab_size=50):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.classes = 5  # 5 different categories
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random sequence of integers representing word indices
        seq = torch.randint(0, self.vocab_size, (self.seq_length,))
        label = torch.randint(0, self.classes, (1,))
        return seq, label

# Step 2: Define a model that uses the SimpleAttention mechanism
class AttentionTextClassifier(nn.Module):
    def __init__(self, vocab_size, input_dim, attention_dim, num_classes):
        super(AttentionTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.attention = SimpleAttention(input_dim, attention_dim)
        self.fc = nn.Linear(attention_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, input_dim)
        
        # Pass through attention mechanism
        attention_output, attention_weights = self.attention(embedded)
        
        # Use the output of attention (we can pool it if needed)
        # Here we use the last element's output as a simple representation
        final_output = attention_output[:, -1, :]  # (batch_size, attention_dim)
        
        # Classify
        logits = self.fc(final_output)  # (batch_size, num_classes)
        return logits

# Step 3: Training the model
def train(model, dataloader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (seq, label) in enumerate(dataloader):
            # Forward pass
            optimizer.zero_grad()
            output = model(seq)  # (batch_size, num_classes)
            
            # Compute loss
            loss = criterion(output, label.squeeze())
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

# Step 4: Define hyperparameters and initialize everything
vocab_size = 50
input_dim = 16  # Dimensionality of word embeddings
attention_dim = 32  # Attention dimension
num_classes = 5
batch_size = 32
epochs = 5

# Initialize dataset, dataloader, model, loss function, and optimizer
dataset = RandomTextDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = AttentionTextClassifier(vocab_size, input_dim, attention_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(model, dataloader, optimizer, criterion, epochs)
