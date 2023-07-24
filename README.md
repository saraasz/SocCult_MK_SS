# Markov Poisson-Multinomial Model for Simulating Elections

This repo contains the code for our (Márton Kardos and Sára Szabó) SocCult exam at the Cognitive Science Bsc in Aarhus.
One can replicate our results by running the code in this repository.

## Getting Started

You should get a clean Python environment (ideally python >= 3.10) and install all requirements by running:
The code was run on Ubuntu 20.04 LTS, in any case we recommend that you run these scripts in a Debian-based environment.

```bash
pip install -r requirements.txt
```

## Running Simulations

You can simulate elections with the `simulate.py` script.
This script accepts a configuration file that contains simulation hyperparameters as its one required command line argument.
We provide a config file in the `configs/` directory which we used in the paper, but you can freely vary starting values by
passing in your own config file.

Here's how you can reproduce our results:
```bash
python3 simulate.py configs/simulation.cfg
```

This is the basic anatomy of a config file:

```
[dest]
dir = "simulations"

[simulation]
n_draws = 1000
n_warmup = 1000
n_chains = 4
seed = 0

[parameters]
[parameters.init]
n_voters = 10000
n_candidates = 10
n_dimensions = 10

[parameters.certainty]
start = 0.0
stop = 10.0
num = 11

[parameters.faithfulness]
start = 0.0
stop = 10.0
num = 11

[parameters.tactic]
start = 0.0
stop = 10.0
num = 11
```

This will save the results of each simulation to a `.npz` file in the `simulations/` folder containing the simulation parameters and the election outcomes on all chains.
Initial state will be saved in the `init.npz` file.

## Evaluating Results
You can evaluate the results of the simulated elections according to the scheme outlined in our paper by running the three scripts:
```bash
python3 evaluate_regression_metrics.py # For regression metrics
python3 evaluate_entropy_kld.py # For evaluation as a probability distribution
python3 evaluate_convergence.py # For evaluation as MCMC sampling
```

All scripts will save results in the `results/` folder as `.feather` files of dataframes for fast loading and saving.

## Producing Figures and tables
You can produce the tables and figures in the paper using the following scripts:

```bash
python3 produce_figures.py
python3 produce_tables.py
```
These will save results into the `figures/` and `tables/` directory.
