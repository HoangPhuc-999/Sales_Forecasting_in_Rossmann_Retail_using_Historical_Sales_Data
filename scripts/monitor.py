from __future__ import annotations

import argparse
import json

from rossmann_mlops.monitoring import run_monitoring


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data drift and performance monitoring")
    parser.add_argument("--reference", default="data/train.csv", help="Reference CSV path")
    parser.add_argument("--current", default="data/test.csv", help="Current CSV path to compare against reference")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    report = run_monitoring(args.reference, args.current, args.config)
    print(json.dumps({
        "timestamp": report.timestamp,
        "drift": [result.__dict__ for result in report.drift],
        "performance": report.performance,
        "alert": report.alert,
    }, indent=2))


if __name__ == "__main__":
    main()
