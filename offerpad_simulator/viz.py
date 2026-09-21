"""All figures for the walkthrough. Each function saves one PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
PAL = {"truth": "#222222", "causal": "#1f77b4", "naive": "#d62728", "other": "#ff7f0e"}


def _color(name: str) -> str:
    n = name.lower()
    if n.startswith("truth"):
        return PAL["truth"]
    if "naive" in n or "invalid" in n or "dml" in n:
        return PAL["naive"]
    if "ignores" in n:
        return PAL["other"]
    return PAL["causal"]


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- 1 valuation
def fig_valuation(ovm, test: pd.DataFrame, out: Path) -> Path:
    p = ovm.predict(test)
    y = test["sale_price"].values
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].scatter(y / 1e3, p["v_hat"] / 1e3, s=3, alpha=0.25)
    lim = [y.min() / 1e3, np.percentile(y, 99.5) / 1e3]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(xlim=lim, ylim=lim, xlabel="actual sale price ($k)", ylabel="OVM estimate ($k)", title="OVM point estimate")
    ape_h = np.abs(p["v_hedonic"] / y - 1) * 100
    ape_g = np.abs(p["v_hat"] / y - 1) * 100
    bins = np.linspace(0, 20, 41)
    ax[1].hist(ape_h, bins, alpha=0.55, label=f"hedonic OLS (median {np.median(ape_h):.1f}%)")
    ax[1].hist(ape_g, bins, alpha=0.55, label=f"gradient boosting (median {np.median(ape_g):.1f}%)")
    ax[1].set(xlabel="absolute % error", ylabel="homes", title="Error distribution")
    ax[1].legend()
    err = np.abs(np.log(p["v_hat"] / y)) * 100
    q = pd.qcut(p["unc_halfwidth"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    cover = pd.Series((y >= p["v_lo"]) & (y <= p["v_hi"])).groupby(q.values, observed=True).mean()
    med = pd.Series(err).groupby(q.values, observed=True).median()
    ax[2].bar(med.index.astype(str), med.values, color="#9ecae1")
    for i, (m, c) in enumerate(zip(med.values, cover.values)):
        ax[2].text(i, m + 0.05, f"cov {c:.0%}", ha="center", fontsize=9)
    ax[2].set(xlabel="predicted-uncertainty quintile", ylabel="median |log error| (%)",
              title="Uncertainty is informative -> wider spread where unsure")
    return _save(fig, out, "fig01_valuation.png")


# --------------------------------------------------------------------------- 2 confounding
def fig_confounding(df: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    obs = df[df["in_experiment"] == 0].copy()
    obs["bin"] = pd.qcut(obs["spread"], 15)
    g = obs.groupby("bin", observed=True).agg(s=("spread", "mean"), a=("accept", "mean"))
    ax[0].plot(g["s"] * 100, g["a"] * 100, "o-", color=PAL["naive"])
    ax[0].set(xlabel="spread (%)", ylabel="acceptance (%)", title="Observational: wider spread, SAME acceptance?!")
    e = df[df["in_experiment"] == 1].groupby("exp_offset")["accept"].agg(["mean", "size"])
    se = np.sqrt(e["mean"] * (1 - e["mean"]) / e["size"])
    ax[1].errorbar(e.index * 100, e["mean"] * 100, yerr=1.96 * se * 100, fmt="o-", color=PAL["causal"], capsize=4)
    ax[1].set(xlabel="randomized spread offset (pp)", ylabel="acceptance (%)", title="Randomized: clear downward slope")
    # the confounder: analysts cut spread on homes in better condition, which sellers value more
    df = df.assign(ubin=pd.qcut(df["true_U"], 10))
    h = df.groupby("ubin", observed=True).agg(u=("true_U", "mean"), s=("spread", "mean"),
                                              r=("true_log_reservation_net", "mean"), v=("v_hat", "mean"))
    ax2 = ax[2].twinx()
    ax[2].plot(h["u"], h["s"] * 100, "o-", color=PAL["naive"], label="spread (analyst)")
    ax2.plot(h["u"], (h["r"] - np.log(h["v"])) * 100, "s--", color="#555", label="seller net reservation / OVM")
    ax[2].set(xlabel="unobserved condition U (sim only)", ylabel="spread (%)", title="Why: U lowers spread AND raises reservation")
    ax2.set_ylabel("log(reservation / OVM) x100")
    ax[2].legend(loc="upper center"); ax2.legend(loc="lower center")
    return _save(fig, out, "fig02_confounding.png")


# --------------------------------------------------------------------------- 3 estimators
def fig_estimators(est: pd.DataFrame, out: Path, title: str, fname: str, xlabel: str) -> Path:
    e = est.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(e) + 1.5))
    for i, r in e.iterrows():
        ax.errorbar(r["estimate"], i, xerr=1.96 * (r["se"] if np.isfinite(r["se"]) else 0), fmt="o",
                    color=_color(r["estimator"]), capsize=3)
    ax.axvline(e["truth"].iloc[0], color="k", ls="--", lw=1.2, label=f"truth = {e['truth'].iloc[0]:.2f}")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(len(e)), e["estimator"])
    ax.set(xlabel=xlabel, title=title)
    ax.legend(loc="lower right")
    return _save(fig, out, fname)


# --------------------------------------------------------------------------- 4 first stage
def fig_first_stage(df: pd.DataFrame, out: Path, v2_day: int) -> Path:
    from .causal import leave_one_out_leniency
    d = df.assign(z_len=leave_one_out_leniency(df))
    a = d.groupby("analyst_id").agg(z=("z_len", "mean"), s=("spread", "mean"), acc=("accept", "mean"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].scatter(a["z"], a["s"] * 100)
    ax[0].set(xlabel="analyst leniency (leave-one-out, pp)", ylabel="mean spread (%)", title="First stage: analyst leniency")
    ax[1].scatter(a["s"] * 100, a["acc"] * 100, color=PAL["causal"])
    b = np.polyfit(a["s"] * 100, a["acc"] * 100, 1)
    xs = np.linspace(a["s"].min() * 100, a["s"].max() * 100, 10)
    ax[1].plot(xs, np.polyval(b, xs), "--", color=PAL["causal"], label=f"slope {b[0]:.2f} pp/pp")
    ax[1].set(xlabel="analyst mean spread (%)", ylabel="analyst acceptance rate (%)", title="Reduced form across analysts")
    ax[1].legend()
    w = d.assign(week=d["day"] // 7).groupby("week")["spread"].mean() * 100
    ax[2].plot(w.index * 7, w.values, ".", color="#555")
    ax[2].axvline(v2_day, color=PAL["naive"], ls="--", label="algorithm v2 cutover")
    ax[2].set(xlabel="day", ylabel="mean spread (%)", title="RD in time: v2 rollout")
    ax[2].legend()
    return _save(fig, out, "fig04_first_stage.png")


# --------------------------------------------------------------------------- 5 HTE
def fig_hte(seg_iv: pd.DataFrame, forest: pd.DataFrame, out: Path) -> Path:
    m = seg_iv.merge(forest[["urgency", "has_agent_quote", "cate_hat"]], on=["urgency", "has_agent_quote"])
    m["segment"] = m["urgency"] + " urgency\n" + np.where(m["has_agent_quote"] == 1, "has agent quote", "no quote")
    order = ["low", "medium", "high"]
    m = m.sort_values(["urgency", "has_agent_quote"], key=lambda s: s.map({k: i for i, k in enumerate(order)}) if s.name == "urgency" else s)
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.bar(x - 0.27, m["cate_true"], 0.27, color=PAL["truth"], label="truth")
    ax.bar(x, m["iv_hat"], 0.27, yerr=1.96 * m["iv_se"], color=PAL["causal"], capsize=3, label="2SLS within segment")
    ax.bar(x + 0.27, m["cate_hat"], 0.27, color="#6baed6", label="causal forest (shrinks to mean)")
    ax.set_xticks(x, m["segment"], fontsize=9)
    ax.set(ylabel="Δ acceptance (pp) per +1pp spread", title="Heterogeneous price sensitivity -> segment-specific spreads")
    ax.legend()
    return _save(fig, out, "fig05_hte.png")


# --------------------------------------------------------------------------- 6 self-appreciation
def fig_self_appreciation(sa: dict, out: Path) -> Path:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    bm = sa["by_market"]
    ax[0].scatter(bm["past_hpa_12m"] * 100, bm["stated_premium"] * 100, s=80, color=PAL["causal"], label="stated / OVM (observed)")
    ax[0].scatter(bm["past_hpa_12m"] * 100, bm["true_self_apprec"] * 100, s=40, marker="x", color="k", label="true belief / value (sim)")
    for _, r in bm.iterrows():
        ax[0].annotate(f"acc {r['accept_rate']:.0%}", (r["past_hpa_12m"] * 100, r["stated_premium"] * 100),
                       textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax[0].set(xlabel="market trailing 12m HPA (%)", ylabel="seller self-appreciation (%)",
              title=f"Sellers extrapolate: passthrough ≈ {sa['passthrough']:.2f}")
    ax[0].legend(fontsize=8)
    c = sa["accept_curve"]
    ax[1].plot(c["offer_vs_stated"] * 100, c["accept_rate"] * 100, "o-", color=PAL["causal"])
    ax[1].axvline(0, color="grey", lw=0.8)
    ax[1].set(xlabel="log(offer / seller's stated value) x100", ylabel="acceptance (%)",
              title="Conversion is about offer vs SELF-assessed value")
    d = sa["data"]
    for a, lab, col in [(1, "accepted", PAL["naive"]), (0, "rejected", "#888")]:
        ax[2].hist(d.loc[d["accept"] == a, "ovm_error"] * 100, bins=np.linspace(-20, 20, 61), density=True,
                   alpha=0.5, color=col, label=f"{lab}: mean {d.loc[d['accept'] == a, 'ovm_error'].mean() * 100:+.1f}%")
    ax[2].set(xlabel="log(true value / OVM) x100", ylabel="density", title="Adverse selection: accepted homes were over-valued")
    ax[2].legend()
    return _save(fig, out, "fig06_self_appreciation.png")


# --------------------------------------------------------------------------- 7 seller optimization
def fig_seller_opt(res: dict, out: Path) -> Path:
    c, o = res["curves"], res["optimum"].set_index("planner")
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    for name, g in c.groupby("planner", sort=False):
        ls = "-" if name == "truth" else "--"
        ax[0].plot(g["delta_pp"], g["accept_rate"] * 100, ls, color=_color(name), label=name)
        ax[1].plot(g["delta_pp"], g["contribution_per_lead"], ls, color=_color(name), label=name)
        ds = o.loc[name, "delta_star_pp"]
        ax[1].plot(ds, g.set_index("delta_pp").loc[ds, "contribution_per_lead"], "*", ms=14, color=_color(name))
    ax[0].set(xlabel="spread change vs current policy (pp)", ylabel="acceptance (%)", title="Conversion curve")
    ax[1].set(xlabel="spread change vs current policy (pp)", ylabel="E[contribution] per offer ($)",
              title="Objective = P(accept) x margin   (* = chosen optimum)")
    ax[1].set_ylim(c["contribution_per_lead"].min() * 0.9, c.loc[c.planner == "truth", "contribution_per_lead"].max() * 1.6)
    ax[0].legend(fontsize=8)
    return _save(fig, out, "fig07_seller_optimization.png")


# --------------------------------------------------------------------------- 8 buyer / 3% cut
def fig_buyer_opt(res: dict, out: Path) -> Path:
    c, o = res["curves"], res["optimum"].set_index("planner")
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for name, g in c.groupby("planner", sort=False):
        ls = "-" if name == "truth" else "--"
        ax[0].plot(g["rho"], g["p_sold_60d"] * 100, ls, color=_color(name), label=name)
        ax[1].plot(g["rho"], g["exp_dom"], ls, color=_color(name), label=name)
        ax[2].plot(g["rho"], g["contribution"], ls, color=_color(name), label=name)
        r = o.loc[name, "rho_star"]
        ax[2].plot(r, g.set_index("rho").loc[r, "contribution"], "*", ms=14, color=_color(name))
    for a in ax:
        a.axvline(1.0, color="grey", lw=0.8)
        a.axvline(0.97, color="green", lw=1, ls=":")
    ax[0].set(xlabel="list price multiplier (0.97 = cut 3%)", ylabel="P(sold within 60d) (%)", title="Conversion vs list price")
    ax[1].set(xlabel="list price multiplier", ylabel="E[days on market]", title="Speed")
    ax[2].set(xlabel="list price multiplier", ylabel="E[contribution] per home ($)", title="Margin objective (* = optimum)")
    t = c[c.planner == "truth"]["contribution"]
    ax[2].set_ylim(t.min() - 2000, t.max() + 4000)
    ax[0].legend(fontsize=8)
    return _save(fig, out, "fig08_buyer_3pct_cut.png")


# --------------------------------------------------------------------------- 9 frontier
def fig_frontier(fr: dict, target_volume: float, out: Path) -> Path:
    f = fr["frontier"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(f["seg_true_volume"] * 100, f["seg_true_contrib"], "o-", color=PAL["causal"], ms=3, label="segment-specific spreads")
    ax.plot(f["uni_true_volume"] * 100, f["uni_true_contrib"], "s-", color="#888", ms=3, label="one uniform spread change")
    ax.axvline(target_volume * 100, color=PAL["naive"], ls="--", label="volume target")
    for lam in [0, 10_000, 20_000]:
        r = f.iloc[(f["shadow_price"] - lam).abs().argmin()]
        ax.annotate(f"λ=${lam/1e3:.0f}k", (r["seg_true_volume"] * 100, r["seg_true_contrib"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set(xlabel="acquisitions per 100 offers", ylabel="E[contribution] per offer ($)",
           title="Margin–volume frontier (policies chosen by the causal model, scored on truth)")
    ax.legend()
    return _save(fig, out, "fig09_margin_volume_frontier.png")


# --------------------------------------------------------------------------- 10 uncertainty
def fig_uncertainty(u: dict, out: Path) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
    P = u["profit_matrix"]
    lo, mid, hi = np.percentile(P, [5, 50, 95], axis=0)
    ax[0].fill_between(u["grid"], lo, hi, alpha=0.3, color=PAL["causal"], label="90% band over (θ, η) draws")
    ax[0].plot(u["grid"], P.mean(axis=0), color=PAL["causal"], label="expected over uncertainty")
    ax[0].axvline(u["rho_robust"], color="k", ls="--", label=f"robust choice {u['rho_robust']:.3f}")
    ax[0].set(xlabel="list price multiplier", ylabel="E[contribution] per home ($)", title="Profit curve under estimation uncertainty")
    ax[0].legend(fontsize=8)
    ax[1].hist(u["rho_star_draws"], bins=np.arange(u["grid"].min(), u["grid"].max() + 0.0025, 0.0025), color=PAL["causal"])
    ax[1].set(xlabel="optimal multiplier in each draw", ylabel="draws", title="Distribution of the optimal price")
    return _save(fig, out, "fig10_elasticity_uncertainty.png")
