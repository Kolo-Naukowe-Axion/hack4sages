#!/usr/bin/env python3
"""Rebin petitRADTRANS correlated-k opacities for the transmission benchmark."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from molmass import Formula
from petitRADTRANS import Radtrans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prt_transmission_benchmark.constants import SOURCE_SPECIES, WAVELENGTH_MAX_UM, WAVELENGTH_MIN_UM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-data-path", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["pRT_input_data_path"] = str(args.input_data_path.resolve())

    masses = {
        base_species: Formula(base_species).isotope.massnumber
        for base_species in {species.split("_")[0] for species in SOURCE_SPECIES}
    }
    corr_k_path = args.input_data_path / "opacities" / "lines" / "corr_k"

    atmosphere = Radtrans(line_species=list(SOURCE_SPECIES), wlen_bords_micron=[WAVELENGTH_MIN_UM, WAVELENGTH_MAX_UM])
    atmosphere.write_out_rebin(
        args.resolution,
        path=str(corr_k_path),
        species=list(SOURCE_SPECIES),
        masses=masses,
    )


if __name__ == "__main__":
    main()
