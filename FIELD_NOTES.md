# Field Notes — why each modelling choice exists

These are the on-ground observations (army logistics personnel + locals, remote Ladakh
border belt, ~3 weeks, Mar 2026) that drove specific decisions in the code. Kept
deliberately qualitative and small-sample — I asked a handful of people, I did not run a
survey. The point is that these choices came from *being there*, not from a paper.

| What I observed / was told | How it shows up in the project |
|---|---|
| The army's resupply effectively keeps civilian villages stocked too — closures are a **dual-use** problem. | Whole framing/objective (README), not a commercial inventory framing. |
| "When the road shuts, **medicines** are the real emergency — fuel you can ration, food you can stretch." | `criticality_rank` in `field_parameters.csv`; `fill_order` (medicines→fuel→food) in `compute_safety_stock`. |
| Closures are **two different beasts** — winter avalanches vs monsoon landslides/washouts. | Separate avalanche/landslide models and seasons. |
| Deep winter the road is simply **known-shut** for months; the genuine uncertainty is the **shoulder season**. | Calendar hard-closure windows (deterministic) vs weather-modelled shoulder days in `build_labels.py`. |
| Nobody keeps a clean closure log; knowledge is **anecdotal** ("that Feb the Zoji La axis was shut a few days"). | `documented_closures.csv` as small real anchors; honest about no public dataset. |
| Depots are **capacity-limited** — you cannot just stock infinite buffer. | `depot_capacity_units` check in `compute_safety_stock`. |
| Talk of a second forward depot exists but it's **way above anyone I spoke to** to decide. | Depot-NPV demoted to an explicitly *illustrative* follow-on case. |

## Honesty note
I could not GPS-survey terrain or get official closure records — those are real gaps.
What field access *did* give me that a remote build can't: the criticality ordering, the
dual-use framing, and a sanity-check against lived memory (see `FIELD_VALIDATION.md`).
