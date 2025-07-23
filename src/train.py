import wandb
import torch
from src.evaluate import evaluate_model
from src.utils import AngularLoss
import os




def train_model(model, train_loader, val_loader, config):
    """
    Used in main.py
    Function to train the model for image rotation prediction.
    Parameters:
        model: The model to be trained.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset. (test_loader is the same, be careful)
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
    elif lossconfig == "angular":
        loss = AngularLoss()
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
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        model.train() #setting the model to training mode
        total_loss = 0.0
        for batch_index, (original_imgs, rotated_imgs, angle) in enumerate(train_loader): 
            optimizer.zero_grad() #zeroing the gradients

            original_imgs = original_imgs.to(device).float()
            rotated_imgs = rotated_imgs.to(device).float()
            angle = angle.to(device).float()

            model_type = config["model"]["type"]
            if model_type == "siamese":
                inputs = (original_imgs, rotated_imgs)  #for siamese model
            else:
                inputs = torch.cat((original_imgs, rotated_imgs), dim=1)  #channel-wise
            
            if isinstance(inputs, tuple):
                predictions = model(*inputs) #unpacking tuple into separate parts
            else:
                predictions = model(inputs)

            #inputs = (original_imgs, rotated_imgs)
            #if isinstance(inputs, tuple):
            #    predictions = model(*inputs)
            #else:
            #    predictions = model(inputs)
            #predictions = model(original_imgs, rotated_imgs) #using the model to get predictions based on original and rotated images
            

            loss_value = loss(predictions.squeeze(), angle) #calculating the loss between prediction and actual angle

            if torch.isnan(loss_value).any(): #checking for NaN in loss
                raise ValueError("NaN detected in loss")
            
            loss_value.backward()
            optimizer.step()
            total_loss = total_loss + loss_value


            #logging in wandb
            wandb.log({
                "train_loss": loss_value.item(),
                "epoch": epoch
            })

        if config["training"]["loss"] == "angular": #if we are using angular loss, we evaluate the model on the validation set
            cosine_loss, mean_abs_angle_error = evaluate_model(config, model, val_loader) #evaluating the model on the validation set - dependency on evaluate.py
            wandb.log({
                "validation_cosine_loss": cosine_loss,
                "epoch": epoch,
                "mean_abs_angle_error": mean_abs_angle_error
            })
            print(f"Epoch [{epoch+1}/{num_epochs}], Cosine Loss: {cosine_loss:.4f}, Mean Absolute Angle Error: {mean_abs_angle_error:.4f}")
        else:
            validation_accuracy = evaluate_model(config, model, val_loader) #evaluating the model on the validation set - dependency on evaluate.py
            avg_loss = total_loss / len(train_loader)
            wandb.log({
                "validation_accuracy": validation_accuracy,
                "epoch": epoch,
                "avg_loss": avg_loss.item()
            })
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss (MSE): {avg_loss.item():.4f}")
    
    if config["training"].get("save", False):
        os.makedirs("checkpoints", exist_ok=True)  #ensure directory exists
        model_name = config["model"]["name"]
        save_path = f"checkpoints/{model_name}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"✅ Model saved to {save_path}")

    return model #returning the trained model



