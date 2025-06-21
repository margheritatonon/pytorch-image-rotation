import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split


#we wrap the dataset in a class
class LoadedRotatedData(Dataset):
    def __init__(self, path="dataset/copy_rotated_dataset.pt"):
        original_images, rotated_images, angles = torch.load(path)
        self.original_images = original_images
        self.rotated_images = rotated_images
        self.angles = angles

    def __len__(self):
        return len(self.original_images)
    
    def __getitem__(self, index):
        return self.original_images[index], self.rotated_images[index], self.angles[index]
    
def get_dataloaders():
    loaded_dataset = LoadedRotatedData()
    train_size = int(0.8 * len(loaded_dataset))
    test_size = len(loaded_dataset) - train_size
    train_dataset, test_dataset = random_split(loaded_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader