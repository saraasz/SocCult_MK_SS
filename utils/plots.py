import numpy as np
import plotly.graph_objects as go


def plot_popularities_over_time(
    popularities: list[np.ndarray], underlying_popularity: np.ndarray
) -> go.Figure:
    n_elections = len(popularities)
    n_parties = len(underlying_popularity)
    fig = go.Figure()
    for i_election in range(n_elections):
        fig = fig.add_trace(
            go.Bar(
                name=f"Election {i_election}",
                visible=i_election == 0,
                x=np.arange(n_parties),
                y=popularities[i_election],
            )
        )
    fig = fig.add_trace(
        go.Bar(
            x=np.arange(n_parties),
            y=underlying_popularity,
            marker=dict(color="red"),
            name="Underlying true distribution",
            visible=True,
        )
    )
    steps = []
    for i in range(len(fig.data)):
        visibility = [False] * len(fig.data)
        # visibility[-1] = True
        step = dict(
            method="update",
            args=[
                {"visible": visibility},
                {"title": "Slider switched to step: " + str(i)},
            ],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][-1] = True  # Toggle i'th trace to "visible"
        steps.append(step)
    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Window: "},
            pad={"t": 50},
            steps=steps,
        )
    ]
    fig = fig.update_layout(sliders=sliders)
    return fig
