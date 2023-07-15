faithfulness_range = [0.0, 4.0]
tactic_range = [0.0, 1.0]

rng = np.random.default_rng(0)
voters = rng.normal(0, 1, (10_000, 10))
voters, candidates = voters[10:], voters[:10]
n_voters, n_candidates = voters.shape[0], candidates.shape[0]
# distances = pairwise_distances(voters, candidates, metric="cosine")
similarities = cosine_similarity(voters, candidates)
initial_state = rng.multinomial(n_voters, np.ones(n_candidates) / n_candidates)

px.histogram(np.ravel(similarities)).show()

n_simulations = 10
params = np.linspace(0.0, 10.0, n_simulations)
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
            similarities,
            honesty=1,
            faithfulness=p,
            tactic=p,
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
            faithfulness=p,
            tactic=p,
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
