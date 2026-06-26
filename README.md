# Leh-Ladakh Highway Supply Chain Disruption Predictor
**[Live app]** · **[GitHub]**

## Problem
Leh-Ladakh is accessible via two highways (NH1, NH3), both subject to
avalanche and landslide closures that can strand the region for weeks.
India Post and BRO have no data-driven framework for predicting closures
or pre-positioning supplies — this project builds one.

## Architecture
Weather data + Terrain features → [Avalanche model | Landslide model]
→ Closure probabilities → Monte Carlo simulation → Safety stock recommendation
→ NPV analysis → Streamlit dashboard

## Technical highlights
- Two-hazard model architecture (separate avalanche/landslide classifiers)
  matching BRO's operational classification
- F2-score threshold optimisation: recall weighted over precision because
  missing a closure costs more than a false alarm
- Time-based train/test split (2019–2022 train, 2023–2024 test) to prevent
  data leakage in time-series setting
- Monte Carlo inventory sizing: stock levels at 95th-percentile closure
  duration, structurally equivalent to VaR in financial risk management
- NPV/IRR analysis framing depot expansion as a capital budgeting decision

## Data
| Source | Variables | Period |
|--------|-----------|--------|
| Open-Meteo archive API | Temp, snowfall, precipitation, wind | 2019–2024 |
| Hardcoded DEM values | Elevation, slope, aspect (5 waypoints) | Static |
| Synthetic labels | Closure events generated from weather thresholds | 2019–2024 |

**Note on synthetic labels:** Ground truth road closure data is not publicly
available. Labels are generated using weather thresholds from Tiwari et al.
(2021). A production deployment would require a data partnership with BRO.

## Results
- Avalanche model AUC: 0.598
- Landslide model AUC: 0.535
- At P(closure) = 65%: recommend 11 days of stock at Leh main depot
- Depot expansion NPV positive after year 7 at base-case assumptions (sensitivity to risk-reduction assumption — adjustable in dashboard)

## Limitations
1. Synthetic labels introduce known noise — real BRO closure data would
   improve model reliability significantly
2. Constant demand assumption — a production system needs demand forecasting
3. Five hardcoded waypoints — full system would process all road segments

## References
1. Tiwari, Vishwakarma et al. (2021) — avalanche prediction, Leh-Manali highway
2. Duran, Gutierrez & Keskinocak (2011) — emergency supply pre-positioning
3. Open-Meteo API: open-meteo.com
4. SRTM DEM: USGS EarthExplorer

## Running locally
```bash
git clone [repo]
pip install -r requirements.txt
python src/data/fetch_weather.py
python src/data/generate_synthetic.py
python src/data/feature_engineering.py
python src/models/train_classifier.py
streamlit run app/streamlit_app.py
```
