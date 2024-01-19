import torch
from diffusion_model import StepwiseReverseDiffusionNet
import matplotlib.pyplot as plt
import imageio
import os
from util_functs import cosine_scaled_noise_level

# Initialize the neural network and optimizer
model = StepwiseReverseDiffusionNet()

# Check if the model weights file exists and load it
model_weights_path = 'diffusion_model.pth'
model.load_state_dict(torch.load(model_weights_path))
print("Loaded saved model weights.")

reconstructed_image = torch.randn(1, 1, 128, 128)  # Example starting image

images = []
num_iterations = 1000
save_interval = 50  # Save every 50 iterations
temp_folder = 'temp_plots'

# Create a temporary folder to store the images
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)

for i in range(num_iterations):
    # Calculate the current noise level and convert it to a tensor
    current_noise_level = cosine_scaled_noise_level(999 - i)  # Adjusted to start from 0.999 to 0
    noise_level_tensor = torch.tensor([current_noise_level], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Forward pass with the current noise level
    reconstructed_image = model(reconstructed_image, noise_level_tensor)

    if i % save_interval == 0:
        # Convert tensor to numpy array for plotting
        img = reconstructed_image.squeeze().detach().cpu().numpy()

        # Plotting
        plt.figure(figsize=(3, 3))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f'Iteration {i}')
        plt.savefig(f'{temp_folder}/plot_{i}.png')
        plt.close()

        # Append the file name to the images list
        images.append(f'{temp_folder}/plot_{i}.png')

# Save the images as a GIF
imageio.mimsave('./imaging/reconstructed_plots.gif', [imageio.imread(img_file) for img_file in images], fps=2)

# Clean up the temporary images
for img_file in images:
    os.remove(img_file)
os.rmdir(temp_folder)
