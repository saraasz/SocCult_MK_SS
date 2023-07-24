"""Calculates MCMC Convergence metrics based on simulation data."""
import glob
from pathlib import Path
from typing import Callable

import arviz as az
import numpy as np
import pandas as pd
from tqdm import tqdm


def run_mcmc_diagnostic(
    simulation_results: list[dict], diagnostic: Callable
) -> pd.DataFrame:
    records = []
    for sim in tqdm(simulation_results, desc=diagnostic.__name__):
        idata = az.convert_to_inference_data({"elections": sim["elections"]})
        results = diagnostic(idata)
        results = results["elections"].to_numpy()
        record = {
            "certainty": float(sim["certainty"]),
            "faithfulness": float(sim["faithfulness"]),
            "tactic": float(sim["tactic"]),
        }
        for i_candidate, value in enumerate(results):
            record[f"{diagnostic.__name__}_cand_{i_candidate}"] = value
        records.append(record)
    return pd.DataFrame.from_records(records)


def main() -> None:
    print("Loading files")
    files = glob.glob("simulations/sim*.npz")
    simulation_results = [dict(np.load(file)) for file in files]

    # Creating output dir
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    print("Running MCMC diagnostics")
    ess_res = run_mcmc_diagnostic(simulation_results, diagnostic=az.ess)
    ess_res.to_feather(out_dir.joinpath("ess.feather"))
    rhat_res = run_mcmc_diagnostic(simulation_results, diagnostic=az.rhat)
    rhat_res.to_feather(out_dir.joinpath("rhat.feather"))
    print("DONE")


if __name__ == "__main__":
    main()
