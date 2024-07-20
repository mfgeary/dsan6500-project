from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image
import torch

class CreateDataset(Dataset): 

    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.data_paths = [f for f in sorted(os.listdir(root_path)) if f.endswith(".png")]
        self.transform = transform

    def __getitem__(self, idx):
        img_file = self.data_paths[idx]
        label = int(img_file.replace('.png', '').split('_')[-1]) - 1
        img_name = self.data_paths[idx]
        img_path = os.path.join(self.root_path, img_name)
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)

def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_batch = 0
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        mean += images.mean(axis=(0, 2, 3))
        std += images.std(axis=(0, 2, 3))
        num_batch += 1

    mean /= num_batch
    std /= num_batch

    return mean, std

def get_dataloaders(root_path, batch_size=500, img_size=(100, 100), train_ratio=0.8, val_ratio=0.05):
    data = CreateDataset(root_path=root_path, 
                         transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()]))

    all_data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    mean, std = get_mean_std(all_data_loader)

    data = CreateDataset(root_path=root_path, 
                         transform=transforms.Compose([transforms.Resize(img_size), 
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean=mean, std=std)]))

    train_size = int(train_ratio * len(data))
    val_size = int(val_ratio * len(data))
    test_size = len(data) - train_size - val_size

    train, val, test = random_split(data, [train_size, val_size, test_size])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader