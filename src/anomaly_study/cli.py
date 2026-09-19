"""Command line entry points; defaults stay small and CPU-only."""

import argparse
import json
from pathlib import Path

from .experiment import run_synthetic
from .public_data import download_nab, run_public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["synthetic", "download", "public"])
    parser.add_argument("--config", default="configs/quick.json")
    parser.add_argument("--output", default="results/local")
    parser.add_argument("--data", default="data/nab")
    parser.add_argument(
        "--subset", default="realAWSCloudwatch", choices=["realAWSCloudwatch", "realAdExchange"]
    )
    args = parser.parse_args()
    if args.command == "download":
        download_nab(args.data, args.subset)
        return
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.command == "synthetic":
        run_synthetic(config, args.output)
    else:
        run_public(config, args.data, args.output)


if __name__ == "__main__":
    main()
