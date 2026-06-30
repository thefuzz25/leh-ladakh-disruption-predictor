# Leh-Ladakh Highway Supply Resilience — a field-researched exercise
**[Live app]** · **[GitHub]**

> A solo problem-solving exercise that started on the ground, not on a laptop. I grew up
> on my father's army postings and the stories from the Ladakh border belt. When I finally
> spent ~3 weeks there myself, I talked directly with army logistics personnel in some of
> the remotest border areas — and saw the problem first-hand: **when the highways close,
> the supply gap doesn't just hit the army, it hits the civilians the army quietly keeps
> stocked.** This repo is my attempt to reason about that problem quantitatively. It is
> not a product and not scalable as-is; the *problem-finding and the reasoning* are the point.

## The insight (the part a remote analyst would miss)
Leh-Ladakh reaches the rest of India by two highways — NH1 (Srinagar–Leh, via Zoji La)
and NH3 (Manali–Leh). Both shut for months on avalanche/landslide closures. Standing in
those villages, the non-obvious thing I learned by *asking* was the **dual-use** structure:
the army's forward logistics effectively underwrite **civilian** supply resilience. So this
isn't a commercial inventory problem — it's a humanitarian/strategic one. That reframing
drives every modelling choice below.

## What I actually built
Real weather + terrain → seasonal & event closure model → closure probabilities →
Monte Carlo stock sizing (criticality-first) → dashboard. An *illustrative* depot-NPV
case sits on top as a "what next", clearly marked as not my decision to make.

## Honesty about the data (read this first)
There is **no public dataset of daily highway closures** — that's a real constraint, not a
shortcut. So labels are **real-anchored, semi-synthetic**, built in a strict provenance order:
1. **Documented (real):** ~15 specific closure events I compiled from local news / BRO
   advisories (`data/documented_closures.csv`) → deterministic ground-truth `=1`.
2. **Calendar (real):** BRO seasonal core-winter hard-closure windows
   (`data/closure_calendar.csv`) — deep winter is known-shut.
3. **Weather-modelled (logical):** remaining shoulder/open-season days get a closure
   probability from the **real Open-Meteo weather** via published thresholds
   (Tiwari et al. 2021 style). **No random coin-flip noise** — that was removed entirely.

`build_labels.py` prints the exact split each run (e.g. avalanche track: ~28 documented +
~644 calendar + ~816 weather-modelled days).

## Field-grounded assumptions (not textbook guesses)
- `data/field_parameters.csv` holds the demand, holding/shortage costs, **and a
  criticality ranking**, each tagged as *elicited from army logistics personnel, Mar 2026*
  (small-sample, qualitative — stated as such).
- The stock recommendation fills depots in **criticality order — medicines → fuel → food —**
  from what people told me runs out first and is life-critical, not a cost-only ordering.
  (`compute_safety_stock` → `fill_order`.)

## Technical highlights
- Two-hazard split (avalanche Oct–May / landslide Jun–Sep) matching the real seasonal reality.
- Time-based split (2019–2022 train, 2023–2024 test) — no time-series leakage.
- Monte Carlo inventory sizing at the 95th-percentile closure duration (structurally a VaR;
  tail average ≈ Expected Shortfall).
- SHAP used for honest exploratory feature attribution (solo exploration, not productionised ML).

## Results (reported honestly)
- Avalanche model AUC ≈ **0.85** — but note this is *partly* the model learning the real
  seasonal calendar (date is genuinely predictive of seasonal closure). Real, not impressive.
- Landslide model AUC ≈ **0.51** — weak, because real landslide events are sparse. I'd rather
  show this than hide it.
- At ~77% Feb closure probability: ~16 days of pre-positioned stock at Leh main depot.
- Depot-expansion case is NPV-negative at base assumptions → *illustratively* "don't build."

## Limitations (what makes this an exercise, not a system)
1. **One weather station (Leh) for all segments** → the two highways currently show similar
   risk; terrain is the only differentiator. A real system needs per-segment weather.
2. **Documented closures are small-sample/anecdotal** — value is authenticity, not power.
3. **Constant demand**, five hardcoded waypoints, solo 3-week scope. Not scalable as-is.

See `FIELD_NOTES.md` (why each choice) and `FIELD_VALIDATION.md` (model vs. what people
told me actually happened).

## References
1. Tiwari, Vishwakarma et al. (2021) — avalanche prediction, Leh-Manali highway
2. Duran, Gutierrez & Keskinocak (2011) — emergency supply pre-positioning
3. Open-Meteo API: open-meteo.com  ·  4. SRTM DEM: USGS EarthExplorer

## Running locally
```bash
pip install -r requirements-dev.txt          # offline pipeline deps
python3 src/data/fetch_weather.py
python3 src/data/build_labels.py
python3 src/data/feature_engineering.py
python3 src/models/train_classifier.py
python3 src/models/evaluate.py
python3 src/models/precompute_app_probs.py    # writes app/precomputed_probs.csv

pip install -r requirements.txt               # slim app deps
streamlit run app/streamlit_app.py
```
