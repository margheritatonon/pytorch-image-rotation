import wandb
import torch
from src.evaluate import evaluate_model



def train_model(model, train_loader, val_loader, config):
    """
    Used in main.py
    Function to train the model for image rotation prediction.
    Parameters:
        model: The model to be trained.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        config: Configuration dictionary containing training parameters.
    """
    #device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device")
    model.to(device)
    
    #learning rate
    lr = config["training"]["learning_rate"]
    #number of epochs
    num_epochs = config["training"]["num_epochs"]
    #batch size
    batch_size = config["training"]["batch_size"]


    #loss function:
    lossconfig = config["training"]["loss"]
    if lossconfig == "mse":
        loss = torch.nn.MSELoss()
    elif lossconfig == "l1":
        loss = torch.nn.L1Loss()
    #add more loss functions here if needed

    #optimizer:
    optimizer_config = config["training"]["optimizer"]
    if optimizer_config == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_config == "sgd":
        momentum = config["training"]["momentum"] if "momentum" in optimizer_config else 0.9 
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    #add more optimizers here if needed

    #training loop
    for epoch in range(num_epochs):
        model.train() #setting the model to training mode
        total_loss = 0.0
        for batch_index, (original_imgs, rotated_imgs, angle) in enumerate(train_loader): 
            optimizer.zero_grad() #zeroing the gradients

            original_imgs = original_imgs.to(device).float()
            rotated_imgs = rotated_imgs.to(device).float()
            angle = angle.to(device).float()

            predictions = model(original_imgs, rotated_imgs) #using the model to get predictions based on original and rotated images
            loss_value = loss(predictions.squeeze(), angle) #calculating the loss between prediction and actual angle

            if torch.isnan(loss_value).any(): #checking for NaN in loss
                raise ValueError("NaN detected in loss")
            
            loss_value.backward()
            optimizer.step()
            total_loss = total_loss + loss_value

            validation_accuracy = evaluate_model(model, val_loader, device) #evaluating the model on the validation set - dependency on evaluate.py

            #logging in wandb
            wandb.log({
                "train_loss": loss_value.item(),
                "epoch": epoch,
                "validation_accuracy": validation_accuracy
            })
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss.item():.4f}")

    return model #returning the trained model



