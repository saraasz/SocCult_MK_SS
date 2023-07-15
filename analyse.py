import glob
from typing import Callable

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

init = np.load("simulations/init.npz")

files = glob.glob("simulations/sim*.npz")

similarities = init["similarities"]
n_voters, n_candidates = similarities.shape
vote = init["similarities"].argmax(axis=1)
votes = np.zeros_like(similarities)
for i_voter, i_vote in enumerate(vote):
    votes[i_voter, i_vote] = 1
ideal_results = votes.sum(axis=0)

simulation_results = [dict(np.load(file)) for file in files]

list(simulation_results[0].keys())


def evaluate_results(
    simulation_results: list[dict], ideal_results: np.ndarray, metric: Callable
) -> pd.DataFrame:
    records = []
    for sim in simulation_results:
        for election_chain in sim["elections"]:
            for election in election_chain:
                record = {
                    "certainty": float(sim["certainty"]),
                    "faithfulness": float(sim["faithfulness"]),
                    "tactic": float(sim["tactic"]),
                    metric.__name__: metric(ideal_results, election),
                }
                records.append(record)
    return pd.DataFrame.from_records(records)


evaluation = evaluate_results(simulation_results, ideal_results, r2_score)

evaluation.info()

float(simulation_results[0]["certainty"])

model = bmb.Model("r2_score ~ tactic", evaluation)
res = model.fit()

az.summary(res)
