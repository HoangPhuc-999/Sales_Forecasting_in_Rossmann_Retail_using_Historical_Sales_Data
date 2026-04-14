from __future__ import annotations

import argparse
import json

from rossmann_mlops.monitoring import retrain_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the Rossmann model and log performance")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    result = retrain_from_config(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
