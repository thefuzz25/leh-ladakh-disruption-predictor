import numpy as np
import pandas as pd
import numpy_financial as npf

# ── 1. Monte Carlo scenario simulator ────────────────────────────────────────

def simulate_disruption_scenarios(
    daily_closure_probs: np.ndarray,
    n_scenarios: int = 2000,
    seed: int = 42
) -> np.ndarray:
    """
    For each scenario: draw Bernoulli trials using the ML probabilities for each day.
    Return array of shape (n_scenarios,) = max consecutive closure days per scenario.

    Why max consecutive days: this determines how long a depot must be self-sufficient,
    which is the directly relevant quantity for stock sizing.

    Why 2000 scenarios: the 95th-percentile estimate stabilises by ~1500 draws.
    """
    np.random.seed(seed)
    max_consecutive = np.zeros(n_scenarios, dtype=int)

    for i in range(n_scenarios):
        draws = np.random.binomial(1, daily_closure_probs)
        max_run, current_run = 0, 0
        for d in draws:
            if d == 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        max_consecutive[i] = max_run

    return max_consecutive


# ── 2. Safety stock calculator ───────────────────────────────────────────────

def load_field_params(path: str = "data/field_parameters.csv"):
    """
    Load demand / cost / criticality assumptions from FIELD-ELICITED parameters
    (interviews with army logistics personnel, Ladakh, Mar 2026) instead of textbook
    literature guesses. Falls back to hardcoded values if the file is unavailable
    (keeps the standalone smoke test runnable).

    criticality_rank: 1 = most life-critical (filled first). Sourced from testimony on
    what actually runs out first when a road closes — an unconventional, field-grounded
    prioritisation rather than a cost-only ordering.
    """
    fallback_demand = {"medicines_units": 50, "fuel_litres": 200, "food_kg": 500}
    fallback_costs = {
        "holding_per_unit_per_day": {"medicines_units": 2.0, "fuel_litres": 0.5, "food_kg": 0.3},
        "shortage_penalty_per_unit": {"medicines_units": 50.0, "fuel_litres": 5.0, "food_kg": 8.0},
    }
    fallback_crit = {"medicines_units": 1, "fuel_litres": 2, "food_kg": 3}
    try:
        df = pd.read_csv(path)
        demand = {r["item"]: float(r["daily_demand"]) for _, r in df.iterrows()}
        costs = {
            "holding_per_unit_per_day": {r["item"]: float(r["holding_per_unit_per_day"]) for _, r in df.iterrows()},
            "shortage_penalty_per_unit": {r["item"]: float(r["shortage_penalty_per_unit"]) for _, r in df.iterrows()},
        }
        criticality = {r["item"]: int(r["criticality_rank"]) for _, r in df.iterrows()}
        return demand, costs, criticality
    except (FileNotFoundError, KeyError):
        return fallback_demand, fallback_costs, fallback_crit


DEFAULT_DEMAND, DEFAULT_COSTS, CRITICALITY = load_field_params()

def compute_safety_stock(
    daily_closure_probs: np.ndarray,
    service_level: float = 0.95,
    daily_demand: dict = DEFAULT_DEMAND,
    costs: dict = DEFAULT_COSTS,
    depot_capacity_units: float = 10000
) -> dict:
    """
    Given ML closure probabilities, compute recommended safety stock.

    Stock sizing method:
      - Run 2000 Monte Carlo scenarios to get a distribution of closure durations
      - Recommended stock = 95th-percentile closure duration × daily demand
        (this is structurally equivalent to Value-at-Risk in financial risk management)
      - Also compute expected tail (average of scenarios above the 95th percentile)
        analogous to Expected Shortfall / CVaR — a more conservative upper bound

    Returns a dict with stock levels, costs, and whether depot capacity is exceeded.
    """
    scenarios     = simulate_disruption_scenarios(daily_closure_probs)
    stock_days    = float(np.percentile(scenarios, service_level * 100))
    tail_days     = float(np.mean(scenarios[scenarios >= stock_days]))

    stock_by_item = {item: round(stock_days * d, 0) for item, d in daily_demand.items()}
    tail_by_item  = {item: round(tail_days  * d, 0) for item, d in daily_demand.items()}

    holding_cost = sum(
        stock_by_item[item] * costs["holding_per_unit_per_day"][item] * stock_days
        for item in daily_demand
    )
    p_shortfall         = float(np.mean(scenarios > stock_days))
    expected_short_days = float(np.mean(np.maximum(0, scenarios - stock_days)))
    shortage_cost = expected_short_days * sum(
        daily_demand[item] * costs["shortage_penalty_per_unit"][item]
        for item in daily_demand
    )

    total_units = sum(stock_by_item.values())

    # Field-grounded CRITICALITY ordering: which items get pre-positioned FIRST when a
    # depot can't hold everything. From testimony (medicines > fuel > food), not cost-only.
    fill_order = [item for item in sorted(daily_demand, key=lambda i: CRITICALITY.get(i, 99))]

    return {
        "stock_days":              round(stock_days, 1),
        "tail_days":               round(tail_days, 1),
        "stock_by_item":           stock_by_item,
        "tail_by_item":            tail_by_item,
        "criticality":             {item: CRITICALITY.get(item, 99) for item in daily_demand},
        "fill_order":              fill_order,
        "holding_cost_inr":        round(holding_cost, 0),
        "expected_shortage_inr":   round(shortage_cost, 0),
        "total_expected_cost_inr": round(holding_cost + shortage_cost, 0),
        "p_shortfall":             round(p_shortfall, 3),
        "capacity_violated":       total_units > depot_capacity_units,
        "service_level":           service_level
    }


# ── 3. Sensitivity table ─────────────────────────────────────────────────────

def sensitivity_table(
    base_prob: float,
    daily_demand: dict = DEFAULT_DEMAND
) -> pd.DataFrame:
    """
    3x3 table: closure probability (low/base/high) x demand multiplier (low/base/high)
    -> total expected cost (INR).

    Standard consulting deliverable — shows how the recommendation changes
    under different risk and demand assumptions.
    """
    probs   = [round(base_prob * 0.7, 2), base_prob, round(base_prob * 1.3, 2)]
    demands = [0.8, 1.0, 1.2]

    rows = {}
    for p in probs:
        row = {}
        probs_arr = np.full(30, p)
        for d in demands:
            scaled = {k: v * d for k, v in daily_demand.items()}
            result = compute_safety_stock(probs_arr, daily_demand=scaled)
            row[f"Demand {d:.0%}"] = f"₹{result['total_expected_cost_inr']:,.0f}"
        rows[f"P(closure) = {p:.0%}"] = row

    return pd.DataFrame(rows).T


# ── 4. Depot NPV analysis ────────────────────────────────────────────────────

def depot_npv(
    current_shortage_cost_inr: float = 1_200_000,
    setup_cost_inr:            float = 2_500_000,
    annual_opex_inr:           float = 300_000,
    risk_reduction_pct:        float = 0.40,
    discount_rate:             float = 0.10,
    horizon_years:             int   = 7
) -> dict:
    """
    ILLUSTRATIVE follow-on business case for a 'should we add a second depot?' decision.
    NOTE: this is a forward-looking what-if, NOT a decision I had the authority or mandate
    to make. It is included to show how the field-observed problem could be escalated into
    a capital-budgeting case — the authentic core deliverable is the stock sizing above.

    Framing: standard capital budgeting (corporate finance).
      Year 0:    -setup_cost (one-time capital outlay)
      Years 1-7:  annual_benefit - annual_opex (operating net cash flow)
      annual_benefit = shortage cost avoided = current_shortage_cost x risk_reduction_pct
      Discount rate = 10% (standard for Indian government infrastructure projects)

    Reports: NPV, IRR, simple payback period.
    """
    annual_benefit = current_shortage_cost_inr * risk_reduction_pct
    annual_ncf     = annual_benefit - annual_opex_inr
    cashflows      = [-setup_cost_inr] + [annual_ncf] * horizon_years

    npv = npf.npv(discount_rate, cashflows)
    irr = npf.irr(cashflows)

    payback = None
    running = -setup_cost_inr
    for yr in range(1, horizon_years + 1):
        running += annual_ncf
        if running >= 0 and payback is None:
            payback = yr

    cf_table = pd.DataFrame({
        "Year":            range(horizon_years + 1),
        "Cash flow (₹)":  [f"({setup_cost_inr:,.0f})"] + [f"{annual_ncf:,.0f}"] * horizon_years,
        "Cumulative (₹)": [
            f"({abs(c):,.0f})" if c < 0 else f"{c:,.0f}"
            for c in np.cumsum(cashflows)
        ]
    })

    return {
        "npv_inr":       round(float(npv), 0),
        "irr_pct":       round(float(irr) * 100, 1) if irr is not None else None,
        "payback_years": payback,
        "annual_benefit_inr": round(annual_benefit, 0),
        "decision":      "INVEST" if npv > 0 else "DO NOT INVEST",
        "cashflow_table": cf_table
    }


if __name__ == "__main__":
    probs = np.full(30, 0.65)
    r = compute_safety_stock(probs)
    print(f"Stock days (95th pct): {r['stock_days']}")
    print(f"Total expected cost:   ₹{r['total_expected_cost_inr']:,.0f}")
    print()
    print(sensitivity_table(0.65))
    print()
    npv_r = depot_npv()
    print(f"NPV: ₹{npv_r['npv_inr']:,.0f} | IRR: {npv_r['irr_pct']}% | Payback: {npv_r['payback_years']} yrs")
    print(f"Decision: {npv_r['decision']}")
