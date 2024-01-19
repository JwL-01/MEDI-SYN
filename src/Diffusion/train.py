import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import optuna
from diffusion_model import StepwiseReverseDiffusionNet
from dataset import DiffusionDataset  # Import the custom dataset class
import torchvision.transforms as transforms
from util_functs import reconstruct_image_iteratively, clear_and_create_directory, save_reconstructed_images  # Import the function

def train_model(train_loader, device, trialObj=None):
    # Use Optuna to suggest a learning rate
    if trialObj is not None:
        lr = trialObj.suggest_float("lr", 1e-6, 1e-3, log=True)
    else:
        print("No trials found, assuming no hyperparameter study...")
        #lr = 0.00054
        #lr = 0.0002944604049665759
        lr = 0.00005

    # Initialize the neural network and optimizer with the trial's learning rate
    model = StepwiseReverseDiffusionNet().to(device)

    # Load saved model weights if they exist
    model_weights_path = './diffusion_model.pth'
    if os.path.exists(model_weights_path) and trialObj is None:
        model.load_state_dict(torch.load(model_weights_path))
        print("Loaded saved model weights.")
    else:
        print("No saved model found")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop variables
    epoch_losses = []
    total_epochs = 70
    save_interval = 1

    # Training loop
    for epoch in range(total_epochs):
        epoch_loss_sum = 0.0
        num_batches = 0

        for batch_idx, (input_image, noise_levels, target_image) in enumerate(train_loader):
            input_image, noise_levels, target_image = input_image.to(device), noise_levels.to(device), target_image.to(device)

            # Forward pass with noise level
            output = model(input_image, noise_levels)
            #loss = torch.nn.SmoothL1Loss()(output, target_image)
            loss = torch.nn.MSELoss()(output, target_image)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            num_batches += 1


            # Print updating progress count
            print(f"\rEpoch {epoch+1}/{total_epochs} - Processing batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}", end="")


        # Save and visualize intermediate results at specified intervals
        if (epoch % save_interval == 0):
            initial_noisy_images = torch.randn(5, 1, 128, 128).to(device)
            reconstructed_images = [reconstruct_image_iteratively(model, initial_noisy_image, 50).cpu().detach().numpy().squeeze(0) for initial_noisy_image in initial_noisy_images] 

            if trialObj is not None:
                save_reconstructed_images(epoch, batch_idx, reconstructed_images, lrate=lr, trialNum=trialObj.number)
            else:
                save_reconstructed_images(epoch, batch_idx, reconstructed_images)

            # Save the model after every epoch
            torch.save(model.state_dict(), f'diffusion_model.pth')
            #print(f"Model saved for Epoch {epoch+1}, batch {batch_idx}+1, and learning rate: {lr}")

        # Calculate average loss for the epoch
        epoch_avg_loss = epoch_loss_sum / num_batches
        print(f"\nEpoch {epoch+1}/{total_epochs} Average Loss: {epoch_avg_loss:.4f}")
        epoch_losses.append(epoch_avg_loss)


    # Plotting the epoch losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, total_epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title("Epoch Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("./training_plots/epoch_losses.png")
    # plt.show() # Uncomment this if you want to display the plo


    # Return the final average loss of the last epoch
    return epoch_losses[-1]

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize and load the dataset
    train_dataset = DiffusionDataset("../data/diffused_train/") # num_images=25,
    train_loader = DataLoader(dataset=train_dataset, batch_size=320, num_workers=8, shuffle=True)

    # Clear plot directory
    clear_and_create_directory("./training_plots/")
    
    finalLoss = train_model(train_loader,device)
    print(f"Loss of last epoch {finalLoss}")

    '''#Learning Rate Optimization Study
    # Setup Optuna study for hyperparameter optimization
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    
    # Run the optimization for a number of trials
    study.optimize(lambda trial: train_model(train_loader, device, trialObj=trial), n_trials=15)

    # Print the results
    print("Study results:")
    print("Best trial:", study.best_trial.params)
    print("Best loss:", study.best_trial.value)'''