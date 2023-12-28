import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Assuming the following classes are defined in their respective files
from prprocess_audio import preprocess_audio
from audiodataset import AudioDataset
from transcription_model import TranscriptionModel
from encode_decode import Encoder, Decoder, Seq2Seq

# Preprocess audio files and save them as JSON
wav_dir = 'HACKATHON_FILES/HACKATHON_FILES/HACKATHON/BENGALI'
json_dir = 'HACKATHON_FILES/HACKATHON_FILE'
preprocess_audio(wav_dir, json_dir)

# Load preprocessed audio data from JSON files
dataset = AudioDataset(json_dir)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the model
encoder = Encoder(input_dim=13, hidden_dim=256, n_layers=2)
decoder = Decoder(output_dim=29, hidden_dim=256, n_layers=2)
model = Seq2Seq(encoder, decoder)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    for i, (inputs, targets) in enumerate(data_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
# ...

# Train the model
for epoch in range(10):
   # ... (training code)

   # Save the trained model
   torch.save(model.state_dict(), 'model.pth')

   # Load the trained model
   model = Seq2Seq(encoder, decoder)
   model.load_state_dict(torch.load('model.pth'))
   model.eval()  # Set the model to evaluation mode

# Load and preprocess the new audio data
new_audio_dir = input('Enter the path to the directory containing the new .wav files: ')
new_json_dir = input('Enter the path to the directory where the preprocessed new data should be saved: ')
preprocess_audio(new_audio_dir, new_json_dir)

# Load preprocessed audio data from JSON files
new_dataset = AudioDataset(new_json_dir)
new_data_loader = DataLoader(new_dataset, batch_size=1, shuffle=False)

# Transcribe the new audio data
for i, (inputs, targets) in enumerate(new_data_loader):
    outputs = model(inputs)
    # The outputs are probably in the form of logits or probabilities.
    # You would need to convert these to text, which depends on how your model and dataset are set up.




    ################## RUN THIS 2