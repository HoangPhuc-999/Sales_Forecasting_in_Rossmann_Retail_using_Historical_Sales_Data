from __future__ import annotations

import argparse
import json

from rossmann_mlops.config import load_config
from rossmann_mlops.train_model import train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full training pipeline: train, evaluate, and save model artifacts")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    result = train_pipeline(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
