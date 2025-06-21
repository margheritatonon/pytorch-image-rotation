import yaml
import argparse

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

    if args.log:
        import wandb
        wandb.init(
            project="image-rotation-prediction",
            config=config)
            

    #starting the process:

    train_loader, val_loader = get_dataloaders(config)
    model = get_model(config['model'])
    #training the model
    trained_model, metrics = train_model(model, train_loader, val_loader, config)
    #evaluating the model
    results = evaluate_model(trained_model, val_loader, config)
    #saving the model and results

if __name__ == "__main__":
    main()