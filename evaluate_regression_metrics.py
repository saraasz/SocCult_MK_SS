"""Calculates regression evaluation metrics to
ideal results based on simulation data."""
import glob
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from tqdm import tqdm


def evaluate_results(
    simulation_results: list[dict],
    ideal_results: np.ndarray,
    metrics: list[Callable],
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
                for metric in metrics:
                    record[metric.__name__] = metric(ideal_results, election)
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

    print("Evaluating simulation results.")
    regression_metrics = evaluate_results(
        simulation_results,
        ideal_results,
        metrics=[
            r2_score,
            max_error,
            mean_absolute_error,
            mean_absolute_percentage_error,
            mean_squared_error,
        ],
    )

    print("Saving results")
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    regression_metrics.to_feather(
        out_dir.joinpath("regression_metrics.feather")
    )


if __name__ == "__main__":
    main()
