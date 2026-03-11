#!/usr/bin/env python3
"""CLI entrypoint for validating the cross-generator biosignature dataset."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossgen_biosignatures.validate_dataset import main


if __name__ == "__main__":
    main()
