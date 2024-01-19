from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import DiffusionDataset
import numpy as np

# Load a subset of your dataset
train_dataset = DiffusionDataset("../data/diffused_train/", num_images=10)
train_loader = DataLoader(dataset=train_dataset, batch_size=300, shuffle=False)
print("Num Batches: ", len(train_loader))

# Initialize lists to store pixel values
pixel_values = []

# Iterate over the dataset and accumulate pixel values
for inputs, _, targets in train_loader:
    pixel_values.extend(inputs.view(-1).numpy())
    pixel_values.extend(targets.view(-1).numpy())

# Convert to a numpy array
pixel_values = np.array(pixel_values)

# Calculate the mean, std deviation, min, and max
mean = np.mean(pixel_values)
std = np.std(pixel_values)
min_val = np.min(pixel_values)
max_val = np.max(pixel_values)

# Print mean, std deviation, min, and max
print("Mean:", mean)
print("Standard Deviation:", std)
print("Minimum Value:", min_val)
print("Maximum Value:", max_val)
