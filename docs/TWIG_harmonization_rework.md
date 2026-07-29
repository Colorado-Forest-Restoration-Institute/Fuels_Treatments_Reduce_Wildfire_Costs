# TWIG interactions rework (branch: `twig-harmonization-rework`)

Reference implementation exploring a fuller rebuild of the TWIG treatment-interactions
summary. **Not merged to `main` by design** — `main` keeps the original single-`type`
classification; this branch is for reference and for the old-vs-new comparison used to
decide how far to go.

## What this branch changes
- `code/02b-TWIG_summary.ipynb` — rebuilt end-to-end:
  - Classifier swapped from the single `type`-column allow-list
    (`trts[trts['type'].isin(trts_keep)]`, which silently dropped null-`type` records)
    to `treatment_interactions.harmonize_twig_activity` — a multi-attribute crosswalk over
    `activity → type → method → twig_category` (canopy-wins → specificity → confidence).
    On the statewide pull: 71,512 resolved / 4,881 dropped (non-fuels) / 9 quarantined
    (all genuine non-treatments; no crosswalk edits needed).
  - Combined variables rebuilt as a **mutually-exclusive area split** to remove the
    collinearity-inducing double-count (see below).
  - Adds a change/sanity report vs the V8 modeling table and a standalone joinable output.
- `code/__functions.py` — appended `combined_treatment_vars` (+ small geometry helpers).

## Why: the double-count / collinearity
In the original `02b`, `All thin`, `Broadcast Burn`, and `Thin + Rx Fire` were each an
independent dissolve/overlay. The thin∩broadcast overlap sits inside all three, so those
acres were counted up to 3×. In the V8 733-fire subset the overlap is ~21% of the thin
signal and ~9% of the burn signal — the mechanical source of the collinearity.

`combined_treatment_vars` partitions each fire's footprint into disjoint pieces
(`thin_only`, `burn_only`, `thin_x_burn`; and a fuels-reduction set B), deriving the
"only" pieces arithmetically (`all_thin − thin_x_burn`) so additivity/monotonicity are
exact (verified sum-error = 0). Headline interactions: `thin_x_burn` = Thin+AllBurn,
`thin_x_fuelred` = Thin+AllBurn+Removal, `thin_x_broadcast` = apples-to-apples successor
to the old `Thin + Rx Fire`.

## Key decisions
- **EPSG:3857 retained** for area/percent to match how `02a` defined the 1 km footprint
  (its `buffer(1000)` in 3857 is ~766 m on the ground, not a true 1 km). Percentages are
  ratios so the Web-Mercator inflation cancels; the `*_acres` columns are therefore
  Web-Mercator values, not true acres. The comparison report converts to true acres via
  an equal-area (EPSG:5070) buffer footprint.
- Post-harmonization burn vocabulary collapses to {Broadcast Burn, Pile Burn}; standalone
  `Removal` is surface removal not tied to a cut (removal-cuts are promoted to Mechanical).
- Sets A and B are alternative model specs — do not use both, or a marginal with its own
  disjoint pieces, in one model. `thin_x_fuelred ⊇ thin_x_burn`.

## Deliverables (regenerable; under `data/` which is gitignored)
- `data/tabular/TWIG_clean_vars.csv` — per-incident cleaned variables (% of buffer), joinable on INCIDENT_ID.
- `data/tabular/qa/twig_harmonize_report.csv` — resolved/dropped/quarantined audit.
- `data/tabular/qa/TWIG_733_comparison_*.{csv,md}` — old-vs-new on the V8 733-fire subset (acres + per-fire).
- `…/xlsx/V8 … (+TWIG_clean_v2).xlsx` — V8 workbook copy with a new `TWIG_clean_v2` sheet (FullMerge untouched).

## Run
Open `code/02b-TWIG_summary.ipynb` in the project kernel (needs `treatment_interactions`,
`seaborn`, `sklearn`, `openpyxl`, `ipywidgets`) and run top-to-bottom.
