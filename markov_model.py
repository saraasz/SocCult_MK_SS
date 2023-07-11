from math import log

import arviz as az
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.special import softmax
from sklearn.metrics import pairwise_distances
from tqdm import tqdm, trange


def simulate_elections(
    rng: np.random.Generator,
    initial_state: np.ndarray,
    distances: np.ndarray,
    honesty: float = 1,
    tactic: float = 1,
    faithfulness: float = 1,
    n_elections: int = 1000,
    pbar: bool = True,
) -> np.ndarray:
    """
    Parameters
    ----------
    initial_state: ndarray of shape (n_candidates)
        Initial election results.
    distances: ndarray of shape (n_voters, n_candidates)
        Distance of each voter from each candidate.
    honesty: float, default 1
        Importance of distance when voting.
    tactic: float, default 1
        Importance of popularity when voting.
    faithfulness: float, default 1
        Importance of previous vote.

    Returns
    -------
    states: ndarray of shape (n_elections, n_candidates)
        Number of votes for each candidate in each election.
    """
    n_voters, n_candidates = distances.shape
    votes = np.zeros((n_voters, n_candidates))
    state = np.copy(initial_state)
    states = []
    lrange = trange if pbar else range
    for i_election in lrange(n_elections):
        d = -honesty * np.log(distances)
        p = tactic * np.log(state)
        f = votes * faithfulness
        logutility = d + p + f
        probability = softmax(logutility, axis=1)
        votes = rng.multinomial(n=1, pvals=probability)
        state = np.sum(votes, axis=0)
        states.append(state)
    return np.stack(states)


faithfulness_range = [0.0, 4.0]
tactic_range = [0.0, 1.0]

rng = np.random.default_rng(0)
voters = normal_voters(rng, 10_000, 10)
voters, candidates = voters[10:], voters[:10]
n_voters, n_candidates = voters.shape[0], candidates.shape[0]
distances = pairwise_distances(voters, candidates, metric="euclidean")
initial_state = rng.multinomial(n_voters, np.ones(n_candidates) / n_candidates)

n_simulations = 10
params = np.linspace(0.0, 1.0, n_simulations)
n_cols = 4
n_rows = n_simulations // n_cols
if n_simulations % n_cols:
    n_rows += 1
fig = make_subplots(
    cols=n_cols, rows=n_rows, subplot_titles=[f"Param: {p}" for p in params]
)
for i, p in enumerate(tqdm(params)):
    row = (i // n_cols) + 1
    col = (i % n_cols) + 1
    with np.errstate(divide="ignore"):
        elections = simulate_elections(
            rng,
            initial_state,
            distances,
            faithfulness=0.0,
            tactic=p,
            honesty=10,
            pbar=False,
            n_elections=1000,
        )
    subfig = px.line(elections)
    for trace in subfig.data:
        if i:
            trace.showlegend = False
        fig.add_trace(trace, row=row, col=col)
fig.show()


n_chains = 4
n_draws = 2000
n_warmup = 2000
elections = np.empty((n_chains, n_draws, n_candidates))
for i_chain in trange(n_chains):
    with np.errstate(divide="ignore"):
        elections[i_chain, :, :] = simulate_elections(
            rng,
            initial_state,
            distances,
            faithfulness=0.0,
            tactic=0.0,
            honesty=1,
            n_elections=n_draws + n_warmup,
            pbar=False,
        )[n_warmup:]

az.ess(
    az.convert_to_inference_data({"elections": elections})
).elections.to_numpy()

az.rhat(az.convert_to_dataset(elections)).x

px.line(elections[0]).show()

elections.shape

np.savez(
    "perfect.npz", honesty=1, faithfulness=0.0, tactic=0.0, elections=elections
)

px.violin(elections.reshape((n_chains * n_draws, n_candidates))).show()
