"""Calculates entropy and relative entropy to
ideal results based on simulation data."""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm


def calculate_entropies(
    simulation_results: list[dict],
    ideal_results: np.ndarray,
) -> pd.DataFrame:
    records = []
    for sim in tqdm(simulation_results):
        for election_chain in sim["elections"]:
            for election in election_chain:
                record = {
                    "certainty": float(sim["certainty"]),
                    "faithfulness": float(sim["faithfulness"]),
                    "tactic": float(sim["tactic"]),
                }
                record["kl_divergence"] = entropy(election, ideal_results)
                record["entropy"] = entropy(election)
                records.append(record)
    return pd.DataFrame.from_records(records)


def main() -> None:
    print("Loading files")
    init = np.load("simulations/init.npz")
    files = glob.glob("simulations/sim*.npz")
    simulation_results = [dict(np.load(file)) for file in files]

    print("Calculating ideal results")
    similarities = init["similarities"]
    vote = init["similarities"].argmax(axis=1)
    votes = np.zeros_like(similarities)
    for i_voter, i_vote in enumerate(vote):
        votes[i_voter, i_vote] = 1
    ideal_results = votes.sum(axis=0)

    print("Calculating entropy and kl-divergence")
    regression_metrics = calculate_entropies(
        simulation_results,
        ideal_results,
    )

    print("Saving results")
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    regression_metrics.to_feather(out_dir.joinpath("entropy.feather"))


if __name__ == "__main__":
    main()
