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

            outputs = model(original_ims, rotated_ims) #using the model

            if not is_regression:
                predictions = torch.argmax(outputs, dim=1)
            else:
                predictions = outputs.squeeze()

            all_predictions.append(predictions.squeeze().cpu().numpy())
            all_actuals.append(angs.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_actuals = np.concatenate(all_actuals)

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Evaluation completed in {time_taken:.2f} seconds") #not sure if i should print this but need to add in wandb

    if is_regression:
        mean_squared_error = np.mean((all_predictions - all_actuals) ** 2) 
        median_squared_error = np.median((all_predictions - all_actuals) ** 2)
        mean_absolute_error = np.mean(np.abs(all_predictions - all_actuals))
        median_absolute_error = np.median(np.abs(all_predictions - all_actuals))

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

     