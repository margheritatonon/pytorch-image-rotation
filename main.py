import yaml
import argparse
import wandb

from src.data import get_dataloaders
from src.models import get_model
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Train rotation prediction model")
    parser.add_argument(
        "--config",
        type = str,
        default = "config/baseline.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log",
        action = "store_true",
        help="Enable logging with W&B"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    wandb.init(
        project = config.get("project_name", "rotation-prediction"),
        name=config.get("experiment_name", "baseline-run"),
        config=config
    )
            

    #starting the process:
    #loaders
    train_loader, test_loader = get_dataloaders() #you can make this dependent on config and add more parameters to the .yaml file if needed
    #accessing the model
    model = get_model(config)
    #training the model
    trained_model = train_model(model, train_loader, test_loader, config)
    #evaluating the model
    results = evaluate_model(config, trained_model, test_loader)
    #saving the model and results

if __name__ == "__main__":
    main()