import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import numpy as np


#we wrap the dataset in a class
class LoadedRotatedData(Dataset):
    def __init__(self, use_sincosencoding=False, path="dataset/copy_rotated_dataset.pt"):
        original_images, rotated_images, angles = torch.load(path)
        self.original_images = original_images
        self.rotated_images = rotated_images
        self.angles = angles
        self.use_sincosencoding = use_sincosencoding
        

    def __len__(self):
        return len(self.original_images)
    
    def __getitem__(self, index):
        angle_deg = self.angles[index]
        angle_rad = angle_deg * np.pi / 180
        if self.use_sincosencoding: 
            angle = torch.stack([torch.sin(angle_rad), torch.cos(angle_rad)])
        else:
            angle = torch.tensor(angle_deg, dtype=torch.float32)
        return self.original_images[index], self.rotated_images[index], angle
    
def get_dataloaders(config):
    use_sincos = config["model"].get("use_sincos_encoding", False) #here we access whether we use sincos encoding or not
    loaded_dataset = LoadedRotatedData(use_sincosencoding=use_sincos)
    train_size = int(0.8 * len(loaded_dataset))
    test_size = len(loaded_dataset) - train_size
    train_dataset, test_dataset = random_split(loaded_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader