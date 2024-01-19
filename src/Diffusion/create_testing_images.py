import torch
import numpy as np
import imageio
import os
from PIL import Image
from diffusion_model import StepwiseReverseDiffusionNet
from util_functs import reconstruct_image_iteratively

def load_and_get_stats(image_path):
    """
    Load an image and return its mean and standard deviation.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: Mean and standard deviation of the image.
    """
    img = Image.open(image_path).convert('L')  # Convert to grayscale if needed
    img_np = np.array(img)
    return img_np.mean(), img_np.std()

def normalize_image(img, target_mean, target_std):
    """
    Normalize an image to match the target mean and standard deviation.

    Args:
        img (numpy.ndarray): Image to be normalized.
        target_mean (float): Target mean.
        target_std (float): Target standard deviation.

    Returns:
        numpy.ndarray: Normalized image.
    """
    img_mean = img.mean()
    img_std = img.std()
    normalized_img = (img - img_mean) / img_std * target_std + target_mean
    return np.clip(normalized_img, 0, 255).astype(np.uint8)

# Initialize the neural network and optimizer
model = StepwiseReverseDiffusionNet()

# Check if the model weights file exists and load it
model_weights_path = 'src/Diffusion/diffusion_model.pth'
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
print("Loaded saved model weights.")

# Load sample image and get its mean and standard deviation
sample_image_path = 'src/Diffusion/imaging/sample/sample_image.png'
sample_mean, sample_std = load_and_get_stats(sample_image_path)

num_iterations = 250
num_samples = 5
output_folder = 'src/Diffusion/.github/workflows/generated_img/'
#output_folder = './imaging/created_images/'

# Create the output directory if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i in range(num_samples):
    # Load or create your initial noisy image (128x128 pixels)
    initial_noisy_image = torch.randn(1, 128, 128)  # Example random noise
    denoised_image = reconstruct_image_iteratively(model, initial_noisy_image, num_iterations)

    # Convert tensor to numpy array
    img = denoised_image.squeeze().detach().cpu().numpy()

    # Normalize the image to match the sample image's mean and standard deviation
    img_normalized = normalize_image(img, sample_mean, sample_std)

    # Save the image using imageio
    imageio.imwrite(os.path.join(output_folder, f'denoised_image_{i+1}.png'), img_normalized)
