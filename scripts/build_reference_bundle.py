#!/usr/bin/env python3
"""Build the compact ADC2023 reference bundle used during remote generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prt_adc2023_validation.reference_bundle import build_reference_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = build_reference_bundle(args.dataset_root, args.output_path)
    print(output_path)


if __name__ == "__main__":
    main()
