import os
import torch
import json
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def _init_(self, directory):
        self.data = []
        
        for file in os.listdir(directory):
            if file.endswith('.json'):
                with open(os.path.join(directory, file), 'r') as f:
                    self.data.extend(json.load(f))

    def _len_(self):
        return len(self.data)

    def _getitem_(self, idx):
        return torch.tensor(self.data[idx])

# Example usage:
directory = 'HACKATHON_FILES'  # replace with your actual directory path
dataset = AudioDataset(directory)

# Save the dataset to a file
torch.save(dataset, 'datasets/dataset.pth')



###############################  RUN THIS 3