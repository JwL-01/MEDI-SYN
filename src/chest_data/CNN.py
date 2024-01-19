#!/usr/bin/env python
# coding: utf-8


import os
import torch
import shutil
import random
import argparse
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.utils.data
from pathlib import Path
import torch.nn.parallel
import torch.optim as optim
import matplotlib.pyplot as plt
from pdb import set_trace as bp
from IPython.display import HTML
import torchvision.utils as vutils
import torchvision.datasets as dset
import matplotlib.animation as animation
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

# Root directory for dataset
dataroot = "./chest_data/"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 1

# Learning rate for optimizers
lr = 0.001

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1



dataset_path = Path('./Coronahack-Chest-XRay-Dataset')
training_path=Path('./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/')
test_path = Path('./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/')

import pandas as pd
dataset_csv = pd.read_csv('./Chest_xray_Corona_Metadata.csv')


dataset_virus = dataset_csv[dataset_csv['Dataset_type'] == 'TRAIN'][dataset_csv['Label_1_Virus_category'] == 'Virus']
dataset_bacteria = dataset_csv[dataset_csv['Dataset_type'] == 'TRAIN'][dataset_csv['Label_1_Virus_category'] == 'bacteria']
dataset_normal = dataset_csv[dataset_csv['Dataset_type'] == 'TRAIN'][dataset_csv['Label'] == 'Normal']

def classwise_images(data):
    shutil.copyfile(src + data['X_ray_image_name'], dst + data['X_ray_image_name'] )

src = './Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/'

dst = './chest_data/virus/'
dataset_virus.apply(classwise_images, axis = 1)

dst = './chest_data/normal/'
dataset_normal.apply(classwise_images, axis = 1)

dst = './chest_data/bacteria/'
dataset_bacteria.apply(classwise_images, axis = 1)

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root='./chest_data/',
                           transform=transforms.Compose([
                               transforms.Grayscale(num_output_channels=1),
                               transforms.Resize((64,64)),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485], std=[0.229])
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16,
                                         shuffle=True, num_workers=2)

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 3)  # Adjust the output size based on your classification task

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x

# Instantiate the model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (you may want to customize this based on your specific use case)
num_epochs = 20
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}")

# Save the trained model if needed
torch.save(model.state_dict(), 'simple_cnn_model.pth')

def encode_data(data):
    label = None
    if data.Label == "Normal":
        label = 1

    if data.Label_1_Virus_category == 'Virus':
        label = 2

    if data.Label_1_Virus_category == 'bacteria':
        label = 0

    return data.X_ray_image_name, label

test_dataset = pd.DataFrame()

test_dataset['(name, label)'] = dataset_csv[dataset_csv['Dataset_type'] == 'TEST'].apply(encode_data, axis = 1)





# Define the same transform used during training
inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Load the trained model
model = SimpleCNN()
model.load_state_dict(torch.load('./simple_cnn_model.pth'))
model.eval()  # Set the model to evaluation mode

count = 0
for name, label in test_dataset['(name, label)']:
  # Load and preprocess the input image for inference
    input_image_path = f"./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/{name}"  # Replace with the path to your image
    input_image = Image.open(input_image_path).convert("L")  # Convert to grayscale
    input_tensor = inference_transform(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class
    _, predicted_class = torch.max(output, 1)

    # Print the predicted class
    # print("Predicted Class:", predicted_class.item(), "actual label: ",label)
    if predicted_class.item() == label:
        count+=1

Accuracy = (count/test_dataset.shape[0])*100

print("Accuracy: ", Accuracy)
