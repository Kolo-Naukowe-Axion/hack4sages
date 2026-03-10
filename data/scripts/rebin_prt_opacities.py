#!/usr/bin/env python3
"""Rebin petitRADTRANS correlated-k opacities to the R=400 setup used by the papers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from molmass import Formula
from petitRADTRANS import Radtrans


SOURCE_SPECIES = [
    "H2O_HITEMP",
    "CO_all_iso_HITEMP",
    "CH4",
    "NH3",
    "CO2",
    "H2S",
    "VO",
    "TiO_all_Exomol",
    "PH3",
    "Na_allard",
    "K_allard",
]


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

    atmosphere = Radtrans(line_species=SOURCE_SPECIES, wlen_bords_micron=[0.1, 251.0])
    atmosphere.write_out_rebin(
        args.resolution,
        path=str(corr_k_path),
        species=SOURCE_SPECIES,
        masses=masses,
    )


if __name__ == "__main__":
    main()
