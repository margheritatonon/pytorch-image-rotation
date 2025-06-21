import yaml
import argparse

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