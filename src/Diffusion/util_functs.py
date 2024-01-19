import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import shutil


def cosine_scaled_noise_level(noise_level):
    """
    Function to scale a noise level using a cosine pattern based on a specific scale law, inverted to go from 0 to 1.
    
    Args:
    noise_level (int): The noise level, between 0 and 999.
    
    Returns:
    float: The cosine-scaled noise level.
    """
    # Set the values for s, e, and tau according to the desired curve
    s, e, tau = 0.2, 1.0, 1  # Example values, adjust as needed

    # Calculate the cosine-scaled noise level according to the schedule
    v_start = np.cos(s * np.pi / 2) ** (2 * tau)
    v_end = np.cos(e * np.pi / 2) ** (2 * tau)
    t = (999 - int(noise_level)) / 999  # Normalize noise level to [0, 1]
    output = np.cos((t * (e - s) + s) * np.pi / 2) ** (2 * tau)
    output = (output - v_start) / (v_end - v_start)

    return np.clip(output, 0, 1)  # Clipping the output to be within [0, 1]




def create_clean_multi_cross(size=128, thickness=6):
    """
    Function to create a clean pattern with multiple crosses and reduced thickness.
    Args:
    size (int): Size of the square pattern.
    thickness (int): Thickness of the cross lines.

    Returns:
    numpy.ndarray: A square pattern with crosses.
    """
    cross = np.zeros((size, size))
    mid = size // 2
    quarter = size // 4
    three_quarter = 3 * size // 4

    for pos in [quarter, mid, three_quarter]:
        cross[pos - thickness // 2:pos + thickness // 2, :] = 1
        cross[:, pos - thickness // 2:pos + thickness // 2] = 1
    return cross

def diffuse_image(image, noise_level=0.5):
    """
    Function to add noise to an image.
    Args:
    image (torch.Tensor): The image to add noise to.
    noise_level (float): The level of noise to add.

    Returns:
    torch.Tensor: The noisy image.
    """
    return image * (1 - noise_level) + torch.randn_like(image) * noise_level

def diffuse_image_levels_linear(image, levels):
    """
    Function to add noise to an image at multiple linear levels.
    Args:
    image (torch.Tensor): The image to add noise to.
    levels (int): Number of levels of noise to add.

    Returns:
    List[torch.Tensor]: List of images with increasing levels of noise.
    """
    noise_levels = np.linspace(0, 1, levels)
    return [diffuse_image(image, noise_level) for noise_level in noise_levels]

def diffuse_image_levels_exponential(image, levels):
    """
    Function to add noise to an image at multiple exponential levels.
    Args:
    image (torch.Tensor): The image to add noise to.
    levels (int): Number of levels of noise to add.

    Returns:
    List[torch.Tensor]: List of images with increasing levels of noise.
    """
    # Create an array where the last level is exactly 1, and the progression towards it is exponential
    noise_levels = np.logspace(0, 1, num=levels, base=10) - 1
    # Normalize so that the maximum noise level is exactly 1
    noise_levels /= np.max(noise_levels)
    return [diffuse_image(image, noise_level) for noise_level in noise_levels]


def diffuse_image_levels_cosine(image, levels):
    """
    Function to add noise to an image at multiple cosine levels based on a specific scale law.
    Args:
    image (torch.Tensor): The image to add noise to.
    levels (int): Number of levels of noise to add.

    Returns:
    List[torch.Tensor]: List of images with noise levels following a cosine pattern.
    """
    # Set the values for s, e, and tau according to the desired curve
    s, e, tau = 0.2, 1.0, 1  # Example values, adjust as needed

    # Flip the ts variable
    ts = (999 - np.linspace(0, 999, levels)) / 999  # Normalize flipped noise levels to [0, 1]

    # Calculate the cosine-scaled noise levels according to the schedule
    v_start = np.cos(s * np.pi / 2) ** (2 * tau)
    v_end = np.cos(e * np.pi / 2) ** (2 * tau)
    noise_levels = [(v_end - np.cos((t * (e - s) + s) * np.pi / 2) ** (2 * tau)) / (v_end - v_start) for t in ts]

    # Assuming diffuse_image is a function that takes an image and a noise level and returns the diffused image
    return [diffuse_image(image, noise_level) for noise_level in noise_levels]


def diffuse_image_levels_sigmoid(image, levels, k=0.1, offset=0.5):
    """
    Function to add noise to an image at multiple sigmoidal levels.
    Args:
    image (torch.Tensor): The image to add noise to.
    levels (int): Number of levels of noise to add.
    k (float): The steepness of the sigmoid curve.
    offset (float): The offset of the sigmoid curve.

    Returns:
    List[torch.Tensor]: List of images with varying levels of noise.
    """
    noise_levels = [1 / (1 + np.exp(-k * (i - offset * levels))) for i in range(levels)]
    return [diffuse_image(image, noise_level) for noise_level in noise_levels]

def reconstruct_image_iteratively(model, initial_noisy_image, num_iterations):
    """
    Function to reconstruct an image iteratively using a given model, with decreasing noise level.
    Args:
        model (torch.nn.Module): The neural network model to use for reconstruction.
        initial_noisy_image (torch.Tensor): The initial noisy image to start with.
        num_iterations (int): Number of iterations to perform.

    Returns:
        torch.Tensor: The reconstructed image (with the batch dimension).
    """
    reconstructed_image = initial_noisy_image.unsqueeze(0)  # Add batch dimension
    for i in range(num_iterations):
        # Calculate the current noise level
        #current_noise_level = (1000 - i) / 1000
        current_noise_level = cosine_scaled_noise_level(999 - i)

        # Convert to a tensor and add necessary dimensions (batch and channel)
        noise_level_tensor = torch.tensor([current_noise_level], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(reconstructed_image.device)
        # Forward pass with the current noise level
        reconstructed_image = model(reconstructed_image, noise_level_tensor)
    return reconstructed_image

def clear_and_create_directory(directory):
   try:
     files = os.listdir(directory)
     for file in files:
       file_path = os.path.join(directory, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All previous epoch images cleared successfully.")
   except OSError:
     print("Error occurred while deleting epoch images.")


def save_reconstructed_images(epoch, batch_idx, reconstructed_images, lrate=None, trialNum=None):
    plt.figure(figsize=(15, 3))
    for i, img in enumerate(reconstructed_images, 1):
        plt.subplot(1, 5, i)
        # Use vmin and vmax to specify the data range for the colormap
        plt.imshow(img.squeeze(0), cmap='gray')#, vmin=-0.71, vmax=0.67)
        plt.axis('off')
    if trialNum is not None:
        plt.suptitle(f'Reconstructed Images at Epoch {epoch + 1}, Batch {batch_idx} and LR = {lrate}')
        plt.savefig(f'./training_plots/trial{trialNum}_epoch{epoch+1}_batch{batch_idx}.png')
    else:
        plt.suptitle(f'Reconstructed Images at Epoch {epoch + 1}, Batch {batch_idx}')
        plt.savefig(f'./training_plots/epoch{epoch+1}_batch{batch_idx}.png')
    plt.close()




