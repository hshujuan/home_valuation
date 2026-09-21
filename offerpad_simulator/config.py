"""Structural parameters of the simulated iBuyer world.

Every number here is a *ground truth* the estimators in `causal.py` try to
recover. Change them and re-run `run_all.py` to see how conclusions move.
"""
from dataclasses import dataclass, field


@dataclass
class SimConfig:
    # ---------------- scale ----------------
    n_leads: int = 60_000          # seller requests that receive an offer
    n_history: int = 40_000        # historical closed MLS sales used to train the OVM
    n_markets: int = 6
    n_analysts: int = 25           # pricing operators who review offers
    n_days: int = 540              # 18 months of offer activity
    seed: int = 7

    # ---------------- home value ----------------
    beta_condition: float = 0.035  # log-value per 1 sd of UNOBSERVED condition U
    market_effects: tuple = (-0.25, -0.10, 0.00, 0.10, 0.20, 0.30)
    past_hpa_12m: tuple = (0.01, 0.03, 0.05, 0.07, 0.02, 0.09)   # trailing appreciation by market

    # ---------------- offer policy (acquisition) ----------------
    fee: float = 0.05              # service fee
    spread_base: float = 0.035
    spread_per_uncertainty: float = 0.40   # spread widens with OVM interval half-width
    analyst_soft_info: float = 0.010       # spread cut per 1 sd of analyst-perceived condition
    analyst_soft_noise: float = 0.7        # analyst sees U + N(0, this)
    leniency_sd: float = 0.006             # analyst-specific systematic spread shift
    experiment_share: float = 0.35         # leads in the randomized spread test
    experiment_offsets: tuple = (-0.02, -0.01, 0.0, 0.01, 0.02)
    v2_day: int = 300                      # algorithm v2 rolls out on this day
    v2_shift: float = 0.008                # v2 widens spread by 80 bps
    rate_passthrough: float = 0.5          # spread moves 0.5pp per 1pp mortgage-rate deviation

    # ---------------- seller behavior ("self-appreciation") ----------------
    self_apprec_base: float = 0.03         # sellers value home 3% above market on average
    self_apprec_hpa_passthrough: float = 0.6   # ... plus extrapolation of their market's recent HPA
    self_apprec_sd: float = 0.04
    rate_on_reservation: float = -1.5      # EXCLUSION VIOLATION: rates move seller reservation directly
    traditional_sale_cost: float = 0.08    # commissions, prep, double-moving, carrying
    accept_intercept: float = 0.9
    # log-odds of accepting per unit of log(offer / net reservation), by seller type
    k_by_urgency: dict = field(default_factory=lambda: {"low": 30.0, "medium": 22.0, "high": 12.0})
    k_quote_multiplier: float = 1.5        # sellers holding a competing agent quote are more price-sensitive
    convenience_by_urgency: dict = field(default_factory=lambda: {"low": 0.0, "medium": 0.02, "high": 0.05})

    # ---------------- resale (disposition) ----------------
    reno_days: int = 30
    list_soft_info: float = 0.02           # pricing team lists nicer homes higher (sees condition)
    list_demand_response: float = 0.01     # ... and lists 1% higher per 1 sd of local demand heat
    list_experiment_share: float = 0.25
    list_experiment_arms: tuple = (-0.03, 0.0, 0.03)   # randomized list at -3% / as-is / +3%
    weibull_shape: float = 1.4
    dom_scale_days: float = 38.0           # median-ish DOM for a fairly priced home
    theta_price: float = 12.0              # TRUE: log DOM-scale per unit log overpricing
    demand_on_dom: float = 0.25            # hot markets sell faster (log DOM-scale per 1 sd heat)
    concession: float = 0.012              # typical buyer concession off list
    overprice_giveback: float = 0.6        # share of over-pricing (vs TRUE value) negotiated away
    decay_start_day: int = 30              # stale listings take price cuts ...
    decay_per_day: float = 0.0005          # ... ~1.5% per month after day 30
    max_decay: float = 0.10
    selling_cost: float = 0.025            # commissions + closing, % of sale price
    holding_cost_per_day: float = 0.00022  # taxes, insurance, HOA, utilities + cost of capital (~8%/yr of value)
    repair_overrun_sd: float = 0.25
