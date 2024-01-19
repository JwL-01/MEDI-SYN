from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
from util_functs import cosine_scaled_noise_level

class DiffusionDataset(Dataset):
    def __init__(self, directory, num_images=None):
        self.directory = directory

        # Define the normalization and ToTensor transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to PyTorch tensor
            transforms.Normalize(mean=[0.5146433], std=[0.279781*2.576])  # Normalize the image so that 99% of the probability lies within [-1,1] (2.576 stdevs "z" test) 
        ])

        # List all image folders and optionally limit the number
        self.image_folders = [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if os.path.isdir(os.path.join(directory, f))]
        if num_images is not None:
            self.image_folders = self.image_folders[:num_images]

        self.images_info = []

        for folder in self.image_folders:
            image_files = sorted(os.listdir(folder))
            for i in range(len(image_files) - 1):
                input_image_path = os.path.join(folder, image_files[i + 1])
                target_image_path = os.path.join(folder, image_files[i])
                
                noise_level = cosine_scaled_noise_level(image_files[i + 1].split('.')[0])
                
                self.images_info.append((input_image_path, target_image_path, noise_level))

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        input_image_path, target_image_path, noise_level = self.images_info[idx]
        
        input_image = Image.open(input_image_path).convert('L')  # Convert to grayscale
        target_image = Image.open(target_image_path).convert('L')  # Convert to grayscale

        # Apply the transform
        input_image = self.transform(input_image)
        target_image = self.transform(target_image)

        return input_image, torch.tensor([noise_level], dtype=torch.float32), target_image

# Example usage
# mean, std = [0.5146433], [0.279781]  # Use your calculated mean and std
# dataset = DiffusionDataset("../data/diffused_train/", num_images=100)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
