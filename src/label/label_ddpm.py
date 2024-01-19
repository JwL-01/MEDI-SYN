import os
import zipfile
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from label_model import *
from util_functs import *


# Donwnload your dataset here for full dataset https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data
# and edit zip_file_path for your dataset.
# Current dataset is the dataset that contains minimal amount of data that can nearly reproduce the original model.

# Unzipping the dataset
zip_file_path = './src/label/small.zip'  # Name of the zip file
extraction_path = './src/label/extracted_data'  # Folder where contents will be extracted

# Create the extraction directory if it doesn't exist
os.makedirs(extraction_path, exist_ok=True)

# Open and extract the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


# Load dataset
dataset = datasets.ImageFolder(root=os.path.join(extraction_path), transform=transform)

# Number of classes
num_classes = len(dataset.classes)

# Check for CUDA availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA is {'available' if torch.cuda.is_available() else 'not available'}. Using {'GPU' if torch.cuda.is_available() else 'CPU'}.")

# Initialize the neural network and optimizer
model = StepwiseReverseDiffusionNet2(num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Split the dataset into training and validation sets
train_size = 0.8
num_train = int(len(dataset) * train_size)
num_valid = len(dataset) - num_train
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [num_train, num_valid])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)

class CustomChestXrayDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform
        self.images = os.listdir(image_dir)  # List of image file names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = datasets.folder.default_loader(img_path)  # Default loader handles image loading

        label = self.labels[img_name]  # Retrieve the label for the given image

        if self.transform:
            image = self.transform(image)

        return image, label

# Check for CUDA availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA is {'available' if torch.cuda.is_available() else 'not available'}. Using {'GPU' if torch.cuda.is_available() else 'CPU'}.")

# Initialize the neural network and optimizer
model = StepwiseReverseDiffusionNet2(num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)


# Split the dataset into training and validation sets
train_size = 0.8
num_train = int(len(dataset) * train_size)
num_valid = len(dataset) - num_train
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [num_train, num_valid])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)

# Check for CUDA availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA is {'available' if torch.cuda.is_available() else 'not available'}. Using {'GPU' if torch.cuda.is_available() else 'CPU'}.")

# Initialize the neural network and optimizer
model = StepwiseReverseDiffusionNet2(num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00005)
criterion = nn.SmoothL1Loss()  # Mean Squared Error Loss for image reconstruction/generation

# Assuming train_loader and valid_loader are already defined
num_epochs = 20
train_losses = []
valid_losses = []

# Training Loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader):
        # Move data to the appropriate device
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images, labels)

        # Compute loss
        loss = criterion(outputs, images)  # Comparing output images with input images

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Print average training loss per epoch
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, labels)
            loss = criterion(outputs, images)
            valid_loss += loss.item()

    # Print average validation loss per epoch
    avg_valid_loss = valid_loss / len(valid_loader)
    valid_losses.append(avg_valid_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_valid_loss:.4f}")



# Assuming you have a DataLoader named 'train_loader'
dataiter = iter(train_loader)
images, _ = next(dataiter)  # Assuming you don't need the labels here

# Number of images to generate
num_images = 4  # Adjust as needed

# Select a subset of images
selected_images = images[:num_images].to(device)
model.eval()
generated_images = []

with torch.no_grad():
    for i in range(num_images):
        label = i  # Assuming you want to generate an image for each label from 0 to num_images-1
        label_tensor = torch.tensor([label]).to(device)

        # Use the corresponding image from the dataset as the initial image
        initial_image = selected_images[i].unsqueeze(0)

        # Generate the image
        generated_image = model(initial_image, label_tensor)

        # Process and store the generated image
        generated_image = generated_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
        generated_images.append(generated_image)


# Set up the plot
fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

# Display each generated image with its label
for i, img in enumerate(generated_images):
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.title(f"Label: {dataset.classes[i]}")
    plt.axis('off')
    plt.savefig(f'../../.github/workflows/generated_image_{i}.png')  # Save each image separately
    plt.close()  # Close the figure to free memory

# epochs = range(1, num_epochs + 1)

# plt.figure(figsize=(10, 6))
# plt.plot(epochs, train_losses, label='Training Loss')
# plt.plot(epochs, valid_losses, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
