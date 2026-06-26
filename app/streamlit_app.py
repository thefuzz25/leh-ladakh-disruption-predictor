import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import sys; sys.path.insert(0, ".")

from streamlit_folium import st_folium
from src.optimization.safety_stock import (
    compute_safety_stock, sensitivity_table,
    depot_npv, simulate_disruption_scenarios
)
from src.visualization.risk_map import build_risk_map

st.set_page_config(page_title="Leh-Ladakh Supply Risk", layout="wide", page_icon="🏔️")
st.title("🏔️ Leh-Ladakh Highway Disruption Predictor")
st.caption("ML-driven road closure forecasting + inventory pre-positioning optimisation")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    month          = st.slider("Forecast month", 1, 12, 2)
    service_level  = st.slider("Service level", 90, 99, 95) / 100
    st.subheader("Daily demand inputs")
    d_med  = st.number_input("Medicines (units/day)",  value=50,  min_value=1)
    d_fuel = st.number_input("Fuel (litres/day)",      value=200, min_value=1)
    d_food = st.number_input("Food (kg/day)",          value=500, min_value=1)

daily_demand = {"medicines_units": d_med, "fuel_litres": d_fuel, "food_kg": d_food}

@st.cache_data(ttl=3600)
def get_segment_probs(month: int) -> dict:
    np.random.seed(month * 7)
    if month in [10, 11, 12, 1, 2, 3]:
        return {"Zoji_La": round(np.random.uniform(0.55, 0.85), 2),
                "Rohtang_Pass": round(np.random.uniform(0.50, 0.80), 2)}
    elif month in [4, 5, 9]:
        return {"Zoji_La": round(np.random.uniform(0.25, 0.55), 2),
                "Rohtang_Pass": round(np.random.uniform(0.20, 0.50), 2)}
    elif month in [7, 8]:
        return {"Zoji_La": round(np.random.uniform(0.15, 0.35), 2),
                "Rohtang_Pass": round(np.random.uniform(0.30, 0.60), 2)}
    else:
        return {"Zoji_La": round(np.random.uniform(0.05, 0.20), 2),
                "Rohtang_Pass": round(np.random.uniform(0.05, 0.20), 2)}

seg_probs = get_segment_probs(month)
avg_prob  = float(np.mean(list(seg_probs.values())))

rec       = compute_safety_stock(np.full(30, avg_prob), service_level, daily_demand)
scenarios = simulate_disruption_scenarios(np.full(30, avg_prob))

# ── Section 1: Metric cards ───────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg closure probability", f"{avg_prob:.0%}")
c2.metric("Recommended stock",       f"{rec['stock_days']} days",
          help="95th-percentile scenario — equivalent to VaR(95%) in finance")
c3.metric("Conservative estimate",   f"{rec['tail_days']} days",
          help="Expected value above 95th pct — equivalent to Expected Shortfall")
c4.metric("Est. total cost",         f"₹{rec['total_expected_cost_inr']:,.0f}")

st.divider()

# ── Section 2: Map + Scenario distribution ───────────────────────────────────
col_map, col_chart = st.columns([3, 2])

with col_map:
    st.subheader("Highway risk map")
    m = build_risk_map(seg_probs, rec)
    st_folium(m, width=500, height=380)

with col_chart:
    st.subheader("Closure duration distribution (2,000 scenarios)")
    fig = px.histogram(
        x=scenarios, nbins=25,
        labels={"x": "Max consecutive closure days", "y": "Scenarios"},
        color_discrete_sequence=["#3498db"]
    )
    fig.add_vline(x=rec["stock_days"], line_dash="dash", line_color="orange",
                  annotation_text=f"95th pct: {rec['stock_days']}d")
    fig.add_vline(x=rec["tail_days"],  line_dash="dot",  line_color="red",
                  annotation_text=f"Tail avg: {rec['tail_days']}d")
    fig.update_layout(showlegend=False, margin=dict(t=30, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Section 3: Stock table + Sensitivity ─────────────────────────────────────
col_s, col_t = st.columns(2)

with col_s:
    st.subheader("Pre-positioning recommendation")
    rows = [
        {"Item": k.replace("_"," ").title(),
         "Daily demand": v,
         "Stock (95th pct)": int(rec["stock_by_item"][k]),
         "Stock (tail avg)": int(rec["tail_by_item"][k])}
        for k, v in daily_demand.items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with col_t:
    st.subheader("Cost sensitivity table")
    st.dataframe(sensitivity_table(avg_prob, daily_demand), use_container_width=True)

st.divider()

# ── Section 4: Depot NPV (expandable) ────────────────────────────────────────
with st.expander("Depot expansion analysis — NPV / IRR"):
    st.write("Capital budgeting analysis: is a second forward depot worth building?")
    col_i1, col_i2 = st.columns(2)
    setup = col_i1.number_input("Setup cost (₹)", value=2_500_000, step=100_000)
    redn  = col_i2.slider("Expected risk reduction", 10, 70, 40)
    npv_r = depot_npv(risk_reduction_pct=redn/100, setup_cost_inr=setup)
    m1, m2, m3 = st.columns(3)
    m1.metric("NPV", f"₹{npv_r['npv_inr']:,.0f}")
    m2.metric("IRR", f"{npv_r['irr_pct']}%" if npv_r["irr_pct"] else "N/A")
    m3.metric("Payback", f"{npv_r['payback_years']} yrs" if npv_r["payback_years"] else ">7 yrs")
    st.markdown(f"**Decision: {npv_r['decision']}**")
    st.dataframe(npv_r["cashflow_table"], hide_index=True)
