"""Figures consume the same tables as the report and application."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px


def response_figure(curve):
    figure = px.line(curve, x="list_price", y="sale_probability", markers=True,
                     title="Supported resale prices and estimated 30-day completion")
    figure.update_yaxes(tickformat=".0%", range=[0, 1])
    return figure


def profit_figure(curve):
    return px.line(curve, x="list_price",
                   y=["expected_gross_profit", "expected_contribution"], markers=True,
                   title="Lifecycle profit includes unsold inventory continuation")


def save_figures(tables, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    curve = tables["profile_curve"]
    axes[0].plot(curve.list_price, curve.sale_probability, marker="o")
    axes[0].set(xlabel="List price ($)", ylabel="30-day completion probability",
                title="Synthetic supported price response")
    axes[1].plot(curve.list_price, curve.expected_gross_profit, marker="o", label="Gross")
    axes[1].plot(curve.list_price, curve.expected_contribution, marker="o", label="Contribution")
    axes[1].set(xlabel="List price ($)", ylabel="Expected lifecycle profit ($)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(directory / "response_and_profit.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    effects = tables["causal_effects"]
    labels = effects.design + ": " + effects.method
    ax.errorbar(effects.estimate * 100, range(len(effects)),
                xerr=1.96 * effects.se * 100, fmt="o", label="Estimate and 95% interval")
    ax.scatter(effects.oracle_ate * 100, range(len(effects)), marker="x", label="Oracle ATE")
    ax.set_yticks(range(len(effects)), labels)
    ax.set(xlabel="3% cut effect (percentage points)", title="Identification, not just prediction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "causal_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    avm = tables["valuation_predictions"]
    ax.scatter(avm.sale_price, avm.prediction, alpha=0.25, s=10)
    bounds = [avm.sale_price.min(), avm.sale_price.max()]
    ax.plot(bounds, bounds, "--", color="gray")
    ax.set(xlabel="Observed synthetic sale price", ylabel="Predicted value",
           title="Chronological valuation holdout")
    fig.tight_layout()
    fig.savefig(directory / "valuation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    acquisition = tables["acquisition_curves"]
    ax.plot(acquisition.offer_ratio, acquisition.selection_aware_profit, marker="o",
            label="Selection-aware")
    ax.plot(acquisition.offer_ratio, acquisition.independence_approximation, marker="o",
            label="Assume acceptance and margin independent")
    ax.set(xlabel="Offer / AVM", ylabel="Expected contribution per lead ($)",
           title="The accepted pool changes with the offer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "selection.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sensitivity = tables["iv_exclusion_sensitivity"].query("scenario == 'strong'")
    direct = 100 * sensitivity.assumed_direct_effect
    axes[0].plot(direct, 100 * sensitivity.adjusted_effect, marker="o")
    axes[0].fill_between(direct, 100 * sensitivity.lower, 100 * sensitivity.upper, alpha=0.2)
    axes[0].axhline(0, color="gray", linestyle="--")
    axes[0].set(xlabel="Assumed direct recommendation effect (pp)",
                ylabel="Adjusted cut effect (pp)",
                title="Strong IV: conditional Wald intervals")
    frontier = tables["portfolio_frontier"]
    axes[1].plot(100 * frontier.predicted_completion, frontier.predicted_contribution,
                 marker="o", label="Fitted")
    axes[1].plot(100 * frontier.oracle_completion, frontier.oracle_contribution,
                 marker="x", label="Oracle evaluation")
    axes[1].set(xlabel="30-day completion (%)", ylabel="Contribution per home ($)",
                title="Expected-volume versus contribution")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(directory / "design_diagnostics.png", dpi=150)
    plt.close(fig)
