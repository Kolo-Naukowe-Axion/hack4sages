#!/usr/bin/env python3
"""CLI entrypoint for generating an append-compatible TauREx extension bundle."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossgen_biosignatures.tau_extension import main


if __name__ == "__main__":
    main()
