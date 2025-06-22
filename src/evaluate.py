import torch
import numpy as np
import time
import wandb

def evaluate_model(config, model, test_loader):
    """
    Used in main.py
    """
    model.eval()  # Set the model to evaluation mode
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    is_regression = config["model"]["task"] == "regression"

    all_predictions = []
    all_actuals = []
    total_loss = 0.0
    start_time = time.time()
    with torch.no_grad():
        for original_ims, rotated_ims, angs in test_loader:

            original_ims = original_ims.to(device).float()
            rotated_ims = rotated_ims.to(device).float()
            angs = angs.to(device).float()

            if config["model"]["type"] == "siamese":
                inputs = (original_ims, rotated_ims)  # for Siamese model
            else:
                inputs = torch.cat((original_ims, rotated_ims), dim=1)  # channel-wise

            if isinstance(inputs, tuple):
                outputs = model(*inputs)
            else:
                outputs = model(inputs) #using the model

            if not is_regression:
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.append(predictions.squeeze().cpu().numpy())
                all_actuals.append(angs.cpu().numpy())
            elif is_regression:
                if config["model"]["use_sincos_encoding"]:
                    #we decode the sincos encoding back to angles
                    predicted_angles = torch.atan2(outputs[:, 0], outputs[:, 1]) * (180 / np.pi)
                    actual_angles = torch.atan2(angs[:, 0], angs[:, 1]) * (180 / np.pi)
                    predicted_angles = predicted_angles % 360 #normalizing the angles to be between 0 and 360
                    actual_angles = actual_angles % 360
                else:
                    predicted_angles = outputs.squeeze()
                    actual_angles = angs.squeeze()
                all_predictions.append(predicted_angles.cpu().numpy())
                all_actuals.append(actual_angles.cpu().numpy())

            

    all_predictions = np.concatenate(all_predictions)
    all_actuals = np.concatenate(all_actuals)

    end_time = time.time()
    time_taken = end_time - start_time
    

    if is_regression:
        mean_squared_error = np.mean((all_predictions - all_actuals) ** 2) 
        median_squared_error = np.median((all_predictions - all_actuals) ** 2)
        mean_absolute_error = np.mean(np.abs(all_predictions - all_actuals))
        median_absolute_error = np.median(np.abs(all_predictions - all_actuals))
    
        if config["loss"]["angular_loss"]:
            angle_errors = (all_predictions - all_actuals + 180) % 360 - 180 
            absolute_angle_errors = np.abs(angle_errors)
            cosine_loss = 1 - np.cos(np.radians(angle_errors)).mean()
            mean_abs_angle_error = np.mean(absolute_angle_errors)

            wandb.log({
                "mean_squared_error": mean_squared_error,
                "median_squared_error": median_squared_error,
                "mean_absolute_error": mean_absolute_error,
                "median_absolute_error": median_absolute_error,
                "time_taken": time_taken,
                "angle_errors": angle_errors,
                "absolute_angle_errors": absolute_angle_errors,
                "cosine_loss": cosine_loss,
                "mean_abs_angle_error": mean_abs_angle_error
            })
            return cosine_loss, mean_abs_angle_error
        else:
            wandb.log({
                "mean_squared_error": mean_squared_error,
                "median_squared_error": median_squared_error,
                "mean_absolute_error": mean_absolute_error,
                "median_absolute_error": median_absolute_error,
                "time_taken": time_taken
            })
            return mean_squared_error #we only return the MSE here
    else: #classification
        accuracy = np.mean(all_predictions == all_actuals)
        wandb.log({
            "accuracy": accuracy,
            "time_taken": time_taken
        })
        return accuracy

     