"""Produces table in paper."""
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd


def summarize_var(s: pd.Series) -> str:
    median = np.median(s)
    low, high = az.hdi(np.array(s))
    return f"{median:.2f}, ({low:.2f}, {high:.2f})"


def summarize(data: pd.DataFrame, vars: list[str]) -> pd.DataFrame:
    data = data.copy()
    data["Certainty"] = np.where(
        data["certainty"] < 5.0, "low (< 5)", "high (>= 5)"
    )
    data["Tactic"] = np.where(data["tactic"] < 5.0, "low (< 5)", "high (>= 5)")
    data["Loyalty"] = np.where(
        data["loyalty"] < 5.0, "low (< 5)", "high (>= 5)"
    )
    data_summary = data.groupby(["Certainty", "Tactic", "Loyalty"]).agg(
        {var: summarize_var for var in vars}
    )
    return data_summary.T


def main() -> None:
    Path("tables").mkdir(exists_ok=True)
    print("Loading results")
    ess = pd.read_feather("results/ess.feather")
    ess["ess_total"] = ess[[f"ess_cand_{i}" for i in range(10)]].sum(axis=1)
    ess = ess.rename(columns=dict(faithfulness="loyalty"))

    reg = pd.read_feather("results/regression_metrics.feather")
    reg["loyalty"] = reg["faithfulness"]
    reg["MAPE"] = reg["mean_absolute_percentage_error"]

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

    entropy = pd.read_feather("results/entropy.feather")
    entropy = entropy.rename(columns=dict(faithfulness="loyalty"))

    print("Collecting summaries")
    reg_summary = summarize(reg, ["MAPE", "r2_score"]).rename(
        index={"r2_score": "R²"}
    )
    ess_summary = summarize(ess, ["ess_total"]).rename(
        index={"ess_total": "Total Bulk ESS"}
    )
    entropy_summary = summarize(entropy, ["entropy", "kl_divergence"]).rename(
        index={
            "entropy": "Entropy",
            "kl_divergence": "KL Divergence",
        }
    )
    rhat_summary = summarize(rhat, ["rhat"]).rename(index={"rhat": "R hat"})
    print("Joining summary.")
    summary = pd.concat(
        (reg_summary, entropy_summary, ess_summary, rhat_summary), axis=0
    )
    print("Saving.")
    summary.to_html("tables/summary.html")
    print("DONE")
    summary[["low (< 5)"]].to_csv("tables/low_certainty.tsv", sep="\t")
    summary[["high (>= 5)"]].to_csv("tables/high_certainty.tsv", sep="\t")


if __name__ == "__main__":
    main()
