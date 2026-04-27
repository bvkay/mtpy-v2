"""Generate example DecompositionResult files for the documentation
notebook.

Run once locally:
    cd ~/mtpy-v2 && conda run -n strike python \
        docs/source/notebooks/example_results/_generate.py

Produces .pkl files alongside this script for use as fallbacks in
the notebook's real-data section when EDI data is not available.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from loguru import logger

from mtpy.core.transfer_function.z_analysis.decomposition import (
    decompose,
    decompose_each_station,
)
from strike_py.edi_datasets import load_edi_dataset


HERE = Path(__file__).parent


def gen_vulcan() -> None:
    logger.info("Generating Vulcan example results...")
    mts = load_edi_dataset("Vulcan_2022", max_stations=5)
    valid = [
        mt
        for mt in mts
        if mt.Z.z_error is not None
    ]
    if not valid:
        logger.warning("Vulcan: no stations with z_error; skipping")
        return
    result = decompose(valid[0].Z, n_starts=5, realisations=50, seed=42)
    result.save(HERE / "vulcan_single_site.pkl")

    results_per_station = decompose_each_station(
        valid, n_starts=3, seed=42
    )
    with (HERE / "vulcan_per_station.pkl").open("wb") as f:
        pickle.dump(results_per_station, f)


def gen_burra() -> None:
    logger.info("Generating Burra example results...")
    mts = load_edi_dataset("Burra_2017-18", max_stations=5)
    valid = [mt for mt in mts if mt.Z.z_error is not None][:1]
    if not valid:
        logger.warning("Burra: no stations with z_error; skipping")
        return
    result = decompose(valid[0].Z, n_starts=5, realisations=50, seed=42)
    result.save(HERE / "burra_single_site.pkl")


def gen_kalkaroo() -> None:
    logger.info("Generating Kalkaroo example results...")
    mts = load_edi_dataset("Kalkaroo_2022", max_stations=5)
    valid = [mt for mt in mts if mt.Z.z_error is not None][:1]
    if not valid:
        logger.warning("Kalkaroo: no stations with z_error; skipping")
        return
    result = decompose(valid[0].Z, n_starts=5, realisations=50, seed=42)
    result.save(HERE / "kalkaroo_single_site.pkl")


if __name__ == "__main__":
    gen_vulcan()
    gen_burra()
    gen_kalkaroo()
    logger.info(f"Done. Files in {HERE}")
