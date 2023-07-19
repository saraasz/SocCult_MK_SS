from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def produce_heatmap(
    data: pd.DataFrame, measure: str, colorscale="Viridis"
) -> go.Figure:
    data = data.groupby(["tactic", "certainty", "loyalty"])[[measure]].median()
    data = data[measure].unstack(level="loyalty")
    fig = make_subplots(
        cols=4,
        rows=3,
        subplot_titles=[f"Loyalty={i}" for i in range(11)],
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
        x_title="certainty",
        y_title="tactic",
    )
    for i, loyalty in enumerate(np.linspace(0.0, 10.0, 11)):
        group = data[loyalty].unstack()
        trace = go.Heatmap(
            x=np.linspace(0.0, 10.0, 11),
            y=np.linspace(0.0, 10.0, 11),
            z=group.round(2),
            texttemplate="%{z}",
            coloraxis="coloraxis",
        )
        col = (i % 4) + 1
        row = (i // 4) + 1
        fig.add_trace(trace, col=col, row=row)
        fig.update_layout(
            coloraxis_colorbar=dict(title=measure),
            coloraxis_colorscale=colorscale,
        )
    return fig


def main():
    print("Creating output folder.")
    Path("figures").mkdir(exist_ok=True)
    print("Loading regression data.")

    reg = pd.read_feather("results/regression_metrics.feather")
    reg = reg.rename(
        columns=dict(
            faithfulness="loyalty", mean_absolute_percentage_error="MAPE"
        )
    )

    print("Producing MAPE plot.")
    fig = produce_heatmap(reg, "MAPE")
    fig = fig.update_layout(width=1000, height=1000)
    print("Saving.")
    fig.write_image("figures/mape_heatmap.png")

    fig = px.scatter(
        reg[(reg.loyalty == 0)],
        x="tactic",
        y="r2_score",
        facet_col="certainty",
        facet_col_wrap=4,
        category_orders={"certainty": np.linspace(0.0, 10.0, 11)},
        color="r2_score",
        template="plotly_white",
    )
    fig.update_yaxes(range=[-1.0, 1.0])
    fig.update_layout(width=1000, height=1000)
    fig.update_layout(
        coloraxis=dict(
            cmid=0.0,
            cmax=1.0,
            cmin=-1,
            colorscale=px.colors.diverging.Picnic_r,
        )
    )
    fig.write_image("figures/tactic_effect_low_loyalty.png")

    print("Loading ess data.")
    ess = pd.read_feather("results/ess.feather")
    ess["ess_total"] = ess[[f"ess_cand_{i}" for i in range(10)]].sum(axis=1)
    ess = ess.rename(columns=dict(faithfulness="loyalty"))

    print("Producing ESS plot.")
    jess = ess.copy()
    n = len(jess.index)
    jess["tactic"] = jess["tactic"] + np.random.uniform(-0.2, 0.2, n)
    jess["certainty"] = jess["certainty"] + np.random.uniform(-0.2, 0.2, n)
    fig = px.scatter(
        jess,
        x="certainty",
        y="tactic",
        color="loyalty",
        size="ess_total",
        template="plotly_white",
        size_max=40,
        color_continuous_scale="Purpor",
    )
    fig = fig.update_layout(width=1000, height=1000)
    print("Saving.")
    fig.write_image("figures/ess.png")

    print("Loading Rhat data")
    rhat = pd.read_feather("results/rhat.feather")
    rhat = rhat.rename(columns=dict(faithfulness="loyalty"))
    rhat = (
        pd.wide_to_long(
            rhat,
            "rhat_cand_",
            i=["certainty", "loyalty", "tactic"],
            j="candidate",
        )
        .reset_index()
        .rename(columns={"rhat_cand_": "rhat"})
    )
    rhat["R hat"] = np.where(rhat.rhat < 1.1, "< 1.1", ">= 1.1")
    rhat["tactic"] = rhat["tactic"].map(str)
    fig = px.box(
        rhat,
        y="tactic",
        x="loyalty",
        facet_col="certainty",
        facet_col_wrap=4,
        category_orders={"certainty": np.linspace(0.0, 10.0, 11)},
        color="R hat",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig = fig.update_layout(width=800, height=1000)
    print("Saving.")
    fig.write_image("figures/rhat.png")

    print("Box plots for ESS and R hat")
    fig = px.box(
        ess,
        x="loyalty",
        y="ess_total",
        template="plotly_white",
    )
    fig = fig.update_layout(
        width=1000,
        height=1000,
    )
    fig = fig.update_traces(marker_color="rgb(136, 204, 238)")
    fig.write_image("figures/loyalty_ess_box.png")

    fig = px.box(rhat, x="loyalty", y="rhat", template="plotly_white")
    fig = fig.update_layout(width=1000, height=1000)
    fig = fig.update_traces(marker_color="rgb(221, 204, 119)")
    fig.write_image("figures/loyalty_rhat.png")

    print("Loading entropy data")
    entropy = pd.read_feather("results/entropy.feather")
    entropy = entropy.rename(columns=dict(faithfulness="loyalty"))
    print("Producing KL-div plot.")
    fig = produce_heatmap(entropy, "kl_divergence", px.colors.sequential.BuPu)
    fig = fig.update_layout(width=1000, height=1000)
    print("Saving")
    fig.write_image("figures/kl_divergence_heatmap.png")

    print("Producing entropy plot.")
    fig = produce_heatmap(entropy, "entropy", px.colors.sequential.PuBuGn_r)
    fig = fig.update_layout(width=1000, height=1000)
    print("Saving")
    fig.write_image(
        "figures/entropy_heatmap.png",
    )

    print("Producing r squared heatmap")
    fig = produce_heatmap(reg, "r2_score")
    fig = fig.update_layout(
        width=1000,
        height=1000,
        coloraxis=dict(
            cmid=0.0,
            cmax=1.0,
            cmin=-1,
            colorscale=px.colors.diverging.Picnic_r,
        ),
    )
    print("Saving")
    fig.write_image(
        "figures/r2_heatmap.png",
    )

    print("Producing r squared marginal plot.")
    reg["R²"] = ""
    reg["R²"][reg.r2_score <= 0] = "<= 0.0"
    reg["R²"][reg.r2_score > 0.0] = "> 0.0, < 0.5"
    reg["R²"][reg.r2_score > 0.5] = "> 0.5, < 0.75"
    reg["R²"][reg.r2_score > 0.75] = "> 0.75, < 0.9"
    reg["R²"][reg.r2_score > 0.9] = "> 0.9, < 0.95"
    reg["R²"][reg.r2_score > 0.95] = "> 0.95, < 0.99"
    reg["R²"][reg.r2_score > 0.99] = "> 0.99"
    r2_groups = [
        "<= 0.0",
        "> 0.0, < 0.5",
        "> 0.5, < 0.75",
        "> 0.75, < 0.9",
        "> 0.9, < 0.95",
        "> 0.95, < 0.99",
        "> 0.99",
    ]
    params = ["certainty", "tactic", "loyalty"]
    fig = make_subplots(rows=3, cols=1, subplot_titles=params)
    for i, parameter in enumerate(params):
        subfig = px.box(
            reg,
            x=parameter,
            y="R²",
            color="R²",
            category_orders={"R²": r2_groups},
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        for trace in subfig.data:
            if i:
                trace["showlegend"] = False
            fig.add_trace(trace, row=i + 1, col=1)
    fig = fig.update_layout(template="plotly_white")
    fig = fig.update_layout(width=1000, height=1000)
    fig = fig.update_yaxes(showticklabels=False)
    print("Saving")
    fig.write_image("figures/r_squared_marginal.png")

    print("Producing r squared pair plots.")
    subplot_titles = [["", f" (R² {group})", ""] for group in r2_groups]
    subplot_titles = [item for sublist in subplot_titles for item in sublist]
    fig = make_subplots(
        rows=7,
        cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.07,
        vertical_spacing=0.03,
    )
    pairs = [
        ("certainty", "tactic"),
        ("loyalty", "certainty"),
        ("tactic", "loyalty"),
    ]
    scales = ["BuGn", "BuPu", "Blues"]
    for row, group in enumerate(r2_groups):
        for col, (x, y) in enumerate(pairs):
            data = reg[reg["R²"] == group]
            fig.add_trace(
                go.Histogram2d(
                    x=data[x],
                    y=data[y],
                    colorscale=scales[col],
                    showscale=False,
                ),
                col=col + 1,
                row=row + 1,
            )
            fig.update_yaxes(
                title=y, col=col + 1, row=row + 1, title_standoff=0
            )
    fig = fig.update_layout(width=800, height=1600)
    fig = fig.update_layout(margin=dict(b=0, l=0, r=0, t=20))
    fig = fig.update_xaxes(title="certainty", col=1, row=7)
    fig = fig.update_xaxes(title="loyalty", col=2, row=7)
    fig = fig.update_xaxes(title="tactic", col=3, row=7)
    print("Saving")
    fig.write_image("figures/r_squared_pairplots.png")

    print("Loyalty R2 plot")
    fig = px.box(
        reg[reg.tactic == 10],
        x="loyalty",
        y="r2_score",
        template="plotly_white",
        color="certainty",
        category_orders={"certainty": np.linspace(0.0, 10.0, 11)},
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig = fig.update_layout(width=1000, height=1000)
    print("Saving")
    fig.write_image("figures/loyalty_r2.png")

    print("DONE")


if __name__ == "__main__":
    main()
