import numpy as np

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

def diffuse_image_levels_cosine(levels, period = 10):
    """
    Generate noise levels using a cosine-based schedule.
    Args:
    levels (int): Number of levels of noise to generate.
    period (int): Period of the cosine function.

    Returns:
    List[float]: List of noise levels based on the cosine function.
    """
    t = np.linspace(0, np.pi, levels)
    noise_levels = 0.5 * (1 - np.cos((t/np.pi) * period))
    return noise_levels.tolist()


def reconstruct_image_iteratively(model, initial_noisy_image, num_iterations):
    """
    Function to reconstruct an image iteratively using a given model.
    Args:
    model (torch.nn.Module): The neural network model to use for reconstruction.
    initial_noisy_image (torch.Tensor): The initial noisy image to start with.
    num_iterations (int): Number of iterations to perform.

    Returns:
    torch.Tensor: The reconstructed image.
    """
    reconstructed_image = initial_noisy_image.unsqueeze(0)
    for _ in range(num_iterations):
        reconstructed_image = model(reconstructed_image)
    return reconstructed_image.squeeze(0)
