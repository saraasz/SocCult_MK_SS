import argparse
from pathlib import Path

import numpy as np
from confection import Config
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity


def simulate_elections(
    rng: np.random.Generator,
    initial_state: np.ndarray,
    similarities: np.ndarray,
    certainty: float = 1,
    tactic: float = 1,
    faithfulness: float = 1,
    n_elections: int = 1000,
) -> np.ndarray:
    n_voters, n_candidates = similarities.shape
    votes = np.zeros((n_voters, n_candidates))
    state = np.copy(initial_state)
    states = []
    for _ in range(n_elections):
        prob_party = state / np.sum(state)
        logutility = (
            certainty * np.log(similarities + 1)
            + tactic * np.log(prob_party + 1)
            + faithfulness * votes
        )
        probability = softmax(logutility, axis=1)
        votes = rng.multinomial(n=1, pvals=probability)
        state = np.sum(votes, axis=0)
        states.append(state)
    return np.stack(states)


def main():
    # Getting config path as command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    # Parsing config
    config = Config().from_disk(args.config)
    # Creating destination directory
    dest_dir = Path(config["dest"]["dir"])
    dest_dir.mkdir(exist_ok=True, parents=True)
    # Initializing data
    rng = np.random.default_rng(config["simulation"].get("seed", 0))
    init = config["parameters"]["init"]
    voters = rng.normal(0, 1, (init["n_voters"], init["n_dimensions"]))
    candidates = rng.normal(0, 1, (init["n_candidates"], init["n_dimensions"]))
    similarities = cosine_similarity(voters, candidates)
    initial_state = rng.multinomial(
        init["n_voters"], np.ones(init["n_candidates"]) / init["n_candidates"]
    )
    print("Saving initial parameters")
    np.savez(
        Path(dest_dir, "init.npz"),
        voters=voters,
        candidates=candidates,
        similarities=similarities,
        initial_state=initial_state,
    )
    # Creating ranges
    params = config["parameters"]
    certainties = np.linspace(
        params["certainty"]["start"],
        params["certainty"]["stop"],
        params["certainty"]["num"],
    )
    faithfulness = np.linspace(
        params["faithfulness"]["start"],
        params["faithfulness"]["stop"],
        params["faithfulness"]["num"],
    )
    tactic = np.linspace(
        params["tactic"]["start"],
        params["tactic"]["stop"],
        params["tactic"]["num"],
    )
    # Running simulations
    n_chains = config["simulation"]["n_chains"]
    n_draws = config["simulation"]["n_draws"]
    n_warmup = config["simulation"]["n_warmup"]
    for c in certainties:
        for f in faithfulness:
            for t in tactic:
                print(
                    f"Running {n_chains} chains with params: "
                    f"c={c}, f={f}, t={t}"
                )
                elections = np.empty((n_chains, n_draws, init["n_candidates"]))
                for i_chain in range(n_chains):
                    elections[i_chain, :, :] = simulate_elections(
                        rng,
                        initial_state,
                        similarities,
                        certainty=c,
                        faithfulness=f,
                        tactic=t,
                        n_elections=n_draws + n_warmup,
                    )[n_warmup:]
                out_path = Path(config["dest"]["dir"]).joinpath(
                    f"sim_{c}_{f}_{t}.npz"
                )
                print("Saving")
                np.savez(
                    out_path,
                    certainty=c,
                    faithfulness=f,
                    tactic=t,
                    elections=elections,
                )
    print("DONE")


if __name__ == "__main__":
    main()
