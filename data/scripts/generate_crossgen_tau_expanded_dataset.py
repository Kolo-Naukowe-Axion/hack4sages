#!/usr/bin/env python3
"""CLI entrypoint for generating a full expanded cross-generator dataset bundle."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossgen_biosignatures.tau_expanded_dataset import main


if __name__ == "__main__":
    main()
