# PROGRESS

Project 1 — FX implied vol surface engine. **All milestones complete.**

| Milestone | Status | Evidence |
|---|---|---|
| M1 pricing core | done | `fxvol/pricing.py`; parity/known-value/Greek-FD tests (`tests/test_pricing.py`) |
| M2 delta machinery | done | `fxvol/delta.py`; 4-convention round trips, PA branch test, DNS tests (`tests/test_delta.py`) |
| M3 smile calibration | done | `fxvol/calibration.py`; market-strangle reprice + RR exactness (`tests/test_calibration.py`) |
| M4 smile + surface | done | `fxvol/smile/*`, `fxvol/surface.py`; headline round-trip at every pillar (`tests/test_surface.py`) |
| M5 arbitrage | done | `fxvol/arbitrage.py`; clean set passes, frown/calendar violations detected (`tests/test_arbitrage.py`) |
| M6 viz + report + demo | done | `fxvol/viz.py`, `fxvol/io_report.py`, `examples/run_demo.py`, `examples/demo.ipynb` |

Test suite: 209 passed, 2 skipped (PA-call strikes below the turning point —
not on the market branch by construction). Coverage ≈ 94% lines
(`python -m pytest --cov=fxvol`). `ruff check` and `mypy fxvol` clean.

Round-trip acceptance on sample EURUSD data: worst error **1.2e-11 vol
points** across ATM/RR25/BF25/RR10/BF10 at all five pillars (target < 0.01).
Same test passes for a synthetic premium-adjusted USDJPY market including a
2Y forward-delta pillar.

## Convention decisions (with citations)

1. **Premium adjustment is a per-pair registry flag**, set by the premium
   currency being the base ccy (Reiswich–Wystup 2010 §2.2; Clark 2011 §3.3).
   EURUSD/GBPUSD/AUDUSD/NZDUSD unadjusted; USDJPY/USDCHF/USDCAD/USDBRL/
   EURJPY/EURGBP adjusted. Unknown pairs raise — no silent guessing.
2. **Spot vs forward delta cutoff**: spot delta for T ≤ 1Y inclusive, forward
   beyond (Clark 2011 §3.3); configurable per pair
   (`spot_delta_cutoff_years`).
3. **ATM = delta-neutral straddle** by default (RW 2010 §3.2):
   `K = F·e^{+σ²T/2}` unadjusted, `K = F·e^{−σ²T/2}` premium-adjusted (sign
   flip tested). ATMF/ATM-spot selectable.
4. **PA call delta inversion picks the higher-strike branch** after locating
   the turning point `σ√T·N(d2) = n(d2)` (RW 2010 §2.2.2). Unattainable
   deltas raise `ValueError`.
5. **Butterfly = market strangle** (RW 2012 §3): calibration solves the
   2-constraint system (RR difference + strangle-price reprice) with Brent,
   seeded by the naive vols `atm + bf ± rr/2`. The smile-strangle BF
   diagnostic (`WingCalibration.smile_bf`) shows the wedge vs the quoted BF
   (e.g. 1Y EURUSD sample: 1.015% smile vs 1.050% quoted).
6. **Smile interpolation**: PCHIP in (log-moneyness, total variance).
   **Wing extrapolation**: asymptotically flat-in-vol with a C1 exponential
   slope decay (default `extrap_decay` = half node span). Rationale: hard
   flat wings are C0 and their kink produces a genuine negative-density
   spike at the 10Δ strikes — discovered by our own arbitrage checker during
   development and kept as a selectable `"flat"` mode for demonstration.
7. **Term structure**: linear in total variance at constant forward
   log-moneyness (Gatheral 2006); flat-vol extrapolation outside pillars.
   Zero curves linear in T, forwards from covered interest parity.
8. **Day count** act/365F on calendar-day tenor rolls (EOM-clamped month
   adds); no holiday calendar yet — extension point, documented in
   `market_data.add_tenor`.
9. **File units**: percent in CSV/JSON, decimals in code; conversion happens
   only at the IO boundary.

## Known limitations / next steps

* No business-day calendar / cut times (ON pillar = 1 calendar day).
* `mypy fxvol` and `ruff check` both pass clean (config in
  `pyproject.toml`). Python here is 3.10, so 3.10-compatible syntax is used
  (spec asked 3.11+; nothing 3.11-specific is required).
* Arbitrage repair mode intentionally not implemented (diagnostics only, by
  design).
* Project 2 attach point: `TenorCalibration` nodes + `pricing.greeks`
  vega/vanna/volga. Project 3 attach point: `smile/sabr.py` stub.
