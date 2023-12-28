import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# Replace these with your actual data
train_data = torch.randn(100, 10)  # 100 samples, 10 features each
train_targets = torch.randint(0, 2, (100,))  # 100 targets, binary classification
valid_data = torch.randn(20, 10)  # 20 samples, 10 features each
valid_targets = torch.randint(0, 2, (20,))  # 20 targets, binary classification

# Create PyTorch Datasets
train_dataset = TensorDataset(train_data, train_targets)
valid_dataset = TensorDataset(valid_data, valid_targets)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Define a simple model
model = nn.Linear(10, 2)  # Binary classification model

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch in valid_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
        print(f'Epoch {epoch}, Validation loss: {total_loss / len(valid_loader)}')




        ##########################  RUN THIS 5 TH