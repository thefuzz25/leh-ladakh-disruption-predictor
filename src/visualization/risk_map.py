import folium

NH1 = [(34.166, 77.580), (34.218, 75.466), (34.314, 74.856)]
NH3 = [(34.166, 77.580), (32.368, 77.245), (32.241, 77.188)]

DEPOTS = {
    "Leh (main depot)":    (34.166, 77.580),
    "Rohtang (fwd depot)": (32.368, 77.245),
}

def prob_to_color(p: float) -> str:
    if p < 0.30:  return "#27ae60"
    elif p < 0.60: return "#f39c12"
    else:          return "#e74c3c"

def build_risk_map(segment_probs: dict, stock_rec: dict) -> folium.Map:
    """
    segment_probs: {"Zoji_La": 0.72, "Rohtang_Pass": 0.45}
    stock_rec:     result dict from compute_safety_stock()
    """
    m = folium.Map(location=[33.5, 76.5], zoom_start=7, tiles="CartoDB Positron")

    folium.PolyLine(
        NH1, color=prob_to_color(segment_probs.get("Zoji_La", 0.3)),
        weight=5, opacity=0.85,
        tooltip=f"NH1 (Leh–Srinagar) — P(closure): {segment_probs.get('Zoji_La', 0):.0%}"
    ).add_to(m)

    folium.PolyLine(
        NH3, color=prob_to_color(segment_probs.get("Rohtang_Pass", 0.3)),
        weight=5, opacity=0.85,
        tooltip=f"NH3 (Leh–Manali) — P(closure): {segment_probs.get('Rohtang_Pass', 0):.0%}"
    ).add_to(m)

    for name, coords in DEPOTS.items():
        popup_html = f"""
        <b>{name}</b><br>
        Recommended stock: <b>{stock_rec.get('stock_days', '?')} days</b><br>
        95th pct: {stock_rec.get('stock_days','?')}d &nbsp;|&nbsp;
        Expected tail: {stock_rec.get('tail_days','?')}d<br>
        Est. cost: ₹{stock_rec.get('holding_cost_inr', 0):,.0f}
        """
        folium.Marker(
            coords,
            popup=folium.Popup(popup_html, max_width=240),
            icon=folium.Icon(color="blue", icon="home", prefix="fa")
        ).add_to(m)

    return m
