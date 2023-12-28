import torch
from torch import nn
from torch.utils.data import DataLoader

class TranscriptionModel(nn.Module):
    def _init_(self, input_size, hidden_size, output_size, num_layers):
        super(TranscriptionModel, self)._init_()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out.reshape(out.size(0)*out.size(1), -1))
        return out

model = TranscriptionModel(input_size=40, hidden_size=256, output_size=29, num_layers=2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Assuming your data is in inputs and labels
inputs = torch.randn(100, 10, 40)  # replace with your actual data
labels = torch.randint(0, 29, (100, 10))  # replace with your actual data

# Create a DataLoader for your data
dataset = list(zip(inputs, labels))
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.view(-1)  # reshape to (batch_size * sequence_length)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item()}')
            # Save the model
torch.save(model.state_dict(), 'model/model.pth')

############## TO LOS TANSCCRIBE