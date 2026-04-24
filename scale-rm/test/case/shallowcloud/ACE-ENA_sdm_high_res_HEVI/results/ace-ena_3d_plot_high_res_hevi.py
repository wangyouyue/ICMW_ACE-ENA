#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case-specific entry point for ACE-ENA high-resolution HEVI 3-D and DSD animations."""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SHALLOWCLOUD_DIR = SCRIPT_DIR.parents[1]
INPUT_FILENAME = "icmw24_aceena_3d_SCALE-SDM_SDM_NUIST_27000_Yin_35x35x2.5_hevi.nc"
OUTPUT_SUFFIX = "_high_res_hevi"

sys.path.insert(0, str(SHALLOWCLOUD_DIR))

from ace_ena_plotting import run_3d_plots


if __name__ == "__main__":
    run_3d_plots(
        input_filename=INPUT_FILENAME,
        output_suffix=OUTPUT_SUFFIX,
        default_data_dir=SCRIPT_DIR,
    )
