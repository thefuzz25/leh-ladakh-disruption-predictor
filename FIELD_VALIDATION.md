# Field Validation — model output vs. lived testimony

A remote builder can only validate a model against more data. Because I was there, I can
also sanity-check it against **what people told me actually happens**. This is anecdotal,
small-sample validation — not a statistical claim — but it's a check only field access makes
possible.

## Model's monthly closure-risk profile (from `app/precomputed_probs.csv`)
| Month | Model risk | What I was told on the ground | Agreement |
|---|---|---|---|
| Jan | ~0.76 | "Peak shut — avalanches, nobody moves." | ✅ |
| Feb | ~0.77 | "Worst of it; Zoji La axis cleared in bursts." | ✅ |
| Mar | ~0.73 | "Still mostly closed, clearance starting." | ✅ |
| Apr | ~0.10 | "Opening up — this is when convoys resume." | ✅ |
| May–Jul | 0.03–0.05 | "Open season." | ✅ |
| Aug | ~0.17 | "Monsoon landslides hit the Manali side." | ✅ (model picks up a smaller monsoon bump) |
| Sep–Oct | ~0.02–0.06 | "Good window before winter." | ✅ |
| Nov | ~0.08 | "Closing down, depends on first snow." | ⚠️ model reads low; locals say it can shut early some years |
| Dec | ~0.42 | "Going under." | ✅ (transition month) |

## Spot-checks against documented events
- The model's high-risk Jan–Mar window lines up with the real avalanche closures I logged
  in `documented_closures.csv` (e.g. Feb 2021, Feb 2024 Zoji La sector).
- The Aug bump aligns with the monsoon landslide events on NH3 (e.g. Aug 2021, Jul–Aug 2024).

## Where testimony says the model is weak
- **Landslide timing** (AUC ≈ 0.70) improved once labels were keyed to 3-day cumulative
  rainfall, but people still described sudden cloudbursts that a daily feature set only
  partly anticipates — the honest residual weak spot.
- **November early-closure** depends on first-snow timing the monthly average smooths over.

Bottom line: the **seasonal shape** matches lived experience well; the **event-level**
landslide layer is the harder part — and I only know where it falls short because I asked.
