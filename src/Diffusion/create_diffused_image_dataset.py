import os
import torch
from PIL import Image
from util_functs import diffuse_image_levels_cosine
import torchvision.transforms as transforms
from multiprocessing import Pool

def process_image(image_file, source_folder, target_folder):
    print(f"Processing image: {image_file}")

    # Create a subfolder for each image's diffused versions
    image_folder = os.path.join(target_folder, os.path.splitext(image_file)[0])
    os.makedirs(image_folder, exist_ok=True)

    # Load and preprocess the image
    image_path = os.path.join(source_folder, image_file)
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((128, 128))


    # Normalize the image
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0], std=[1.5])  # Normalizing
        
    ])
    image_tensor = transform(image)


    # Generate diffused images
    diffused_images = diffuse_image_levels_cosine(image_tensor, 1000)

    # Save diffused images
    for level, diffused_image in enumerate(diffused_images):
        save_path = os.path.join(image_folder, f"{str(level).zfill(3)}.png")
        diffused_pil = transforms.ToPILImage()(diffused_image.squeeze(0))
        diffused_pil.save(save_path)

    print(f"Finished processing {image_file}")

def create_diffused_dataset(source_folder, target_folder, run_images=None):
    # Ensure the target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Get all jpg files from the source folder
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.png')]
    
    # If run_images is specified, slice the list to that length
    if run_images is not None:
        image_files = image_files[:run_images]

    # Set up multiprocessing
    with Pool(8) as p:
        p.starmap(process_image, [(image_file, source_folder, target_folder) for image_file in image_files])

if __name__ == '__main__':
    create_diffused_dataset('../data/train/', '../data/diffused_train/')
