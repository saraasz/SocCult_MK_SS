from functools import partial
from typing import Callable

import blackjax
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import plotly.express as px


def mixture_logpdf(x, pdf: Callable, params: list[dict]):
    density = 0
    for kwargs in params:
        density += pdf(x["x"], **kwargs)
    return jnp.log(density)


def mixture(pdf: Callable, params: list[dict]) -> Callable:
    return partial(mixture_logpdf, pdf=pdf, params=params)


def sample_logpdf(
    log_pdf: Callable, n: int = 2000, warmup: int = 1000
) -> np.ndarray:
    nuts = blackjax.nuts(
        log_pdf, step_size=1e-2, inverse_mass_matrix=np.array([1])
    )
    rng_key = jax.random.PRNGKey(0)
    state = nuts.init({"x": 0.5})
    step = jax.jit(nuts.step)
    res = np.zeros(n)
    for _ in range(warmup):
        rng_key, nuts_key = jax.random.split(rng_key)
        state, _ = step(nuts_key, state)
    for i in range(n):
        rng_key, nuts_key = jax.random.split(rng_key)
        state, _ = step(nuts_key, state)
        res[i] = state.position["x"].block_until_ready()
    return res


components = [
    # dict(a=4, b=4),
    dict(a=10, b=10),
]
dist = mixture(stats.beta.pdf, params=components)
sample = sample_logpdf(dist, n=10000)
px.histogram(sample).show()

grid = np.arange(0, 1, 0.01)
px.line(x=grid, y=[dist(dict(x=x)) for x in grid]).show()
