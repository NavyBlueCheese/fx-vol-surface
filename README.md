# fxvol — FX Implied Volatility Surface Engine

Production-grade construction of FX implied-volatility surfaces from standard
market quotes (ATM, risk reversals, butterflies), with **first-class FX
convention handling**, **Reiswich–Wystup market-strangle calibration**,
**static no-arbitrage diagnostics**, and a query API for vol / price / Greeks
at any strike, delta and tenor.

The headline acceptance property: **the calibrated surface reprices its own
inputs.** On the shipped sample data the worst round-trip error is
~1e-11 vol points (target: < 0.01 vol points; price round-trips < 1e-9).

```
quotes (ATM / 25Δ RR / 25Δ BF / 10Δ RR / 10Δ BF)
   │  per-pair conventions (spot vs fwd delta, premium-adjusted, DNS ATM)
   ▼
market-strangle calibration (Reiswich–Wystup)      ──►  5 smile nodes / tenor
   ▼
SmileModel (PCHIP total-variance interpolant)      ──►  vol(K) per tenor
   ▼
VolSurface (linear-in-total-variance term interp)  ──►  vol(K,T), vol(Δ,T),
   ▼                                                    price, full Greeks
arbitrage checks (convexity, Durrleman g(k), calendar) + report + plots
```

## Quickstart

```bash
pip install -e .[dev]
python examples/run_demo.py          # full pipeline + report + plots
python -m pytest --cov=fxvol         # test suite (adversarial by design)
```

```python
from fxvol import VolSurface, OptionType, load_csv, check_surface

market  = load_csv("examples/eurusd_quotes.csv")   # vols/rates in percent
surface = VolSurface.from_market(market)           # market-strangle calibration

surface.vol(1.10, 0.5)                             # vol at strike 1.10, T=0.5y
k, v = surface.vol_from_delta(0.25, 0.5, OptionType.CALL)  # 25Δ call point
surface.greeks(k, 0.5, OptionType.CALL)            # delta/gamma/vega/vanna/volga/…

report = check_surface(surface)                    # butterfly + calendar checks
assert report.ok, report.summary()
```

Requires Python 3.10+ (developed and tested on 3.10; only 3.10-compatible
syntax is used).

## Units and quoting conventions (read this first — FX bugs hide here)

* Spot `S` = **domestic (quote) ccy per 1 unit of foreign (base) ccy**
  (EURUSD: EUR foreign, USD domestic).
* Prices are **domestic ccy per 1 unit foreign notional**;
  `fxvol.pricing.convert_premium` converts to %-domestic, %-foreign and
  foreign-pips quotations.
* Vols, rates: decimals in code; **percent in CSV/JSON files** (converted at
  the IO boundary only).
* Rates are continuously compounded; forwards satisfy
  `F = S·exp((r_dom − r_for)·T)`. Given a forward outright instead of
  `r_for`, the foreign rate is implied from covered interest parity.
* Day count for vol time: act/365F (configurable per pair).

## The math

### Pricing — Garman–Kohlhagen (forward form)

`d1 = (ln(F/K) + σ²T/2)/(σ√T)`, `d2 = d1 − σ√T`

`V = φ·df_dom·(F·N(φd1) − K·N(φd2))`, φ=+1 call / −1 put.

Full analytic Greeks (delta spot/forward, gamma, vega, **vanna, volga**,
theta, rho_dom, rho_for) validated against central finite differences —
vanna/volga are load-bearing for Project 2 (vanna-volga).

### Delta conventions (the crux)

| convention          | call delta                  |
|---------------------|-----------------------------|
| forward, unadjusted | `N(d1)`                     |
| spot, unadjusted    | `e^(−r_for·T)·N(d1)`        |
| forward, prem-adj   | `(K/F)·N(d2)`               |
| spot, prem-adj      | `e^(−r_for·T)·(K/F)·N(d2)`  |

* **Premium-adjusted** applies when the premium is paid in the foreign/base
  ccy (per-pair flag in `fxvol.conventions.PAIR_REGISTRY`, following
  Reiswich–Wystup 2010; e.g. EURUSD unadjusted, USDJPY adjusted).
* **Spot vs forward delta**: spot delta up to and including a configurable
  cutoff (default 1Y, Clark 2011 §3.3), forward delta beyond.
* **Premium-adjusted call delta is not monotonic in strike.** The engine
  locates the turning point (`vol√T·N(d2) = n(d2)`) and root-finds on the
  **higher-strike branch**, which is what the market quote refers to. Deltas
  above the attainable maximum raise instead of returning a wrong strike.
  This has a dedicated test that exhibits both roots and asserts the correct
  one is returned.

### ATM

Default **delta-neutral straddle** (DNS): `K_atm = F·e^(+σ²T/2)` unadjusted,
`K_atm = F·e^(−σ²T/2)` premium-adjusted (note the sign flip). ATM-forward and
ATM-spot also supported.

### Butterfly calibration — market strangle vs smile strangle

The quoted `BF` is a **market-strangle (broker fly)** quote, *not* smile
convexity. Per wing level Δ ∈ {25, 10}:

1. `σ_ms = σ_ATM + BF`; strikes `K_c^ms, K_p^ms` at ±Δ using the single vol
   `σ_ms`; **market strangle price**
   `v_ms = Call(K_c^ms, σ_ms) + Put(K_p^ms, σ_ms)` this scalar is the
   actual observable.
2. Solve `(σ_Δc, σ_Δp)` s.t. (a) `σ_Δc − σ_Δp = RR` and (b) the smile
   strangle (strikes at ±Δ with the smile's own vols) reprices `v_ms`.
   1-D Brent with the naive vols `σ_ATM + BF ± RR/2` as seed; convergence
   and reprice accuracy are asserted, never assumed.

The result is a 5-node smile {10Δp, 25Δp, ATM, 25Δc, 10Δc} per tenor whose
implied `(ATM, RR, BF)` round-trip to the inputs at ~1e-11 vol points.

### Smile & term interpolation

* Within tenor: **PCHIP** (shape-preserving monotone cubic) through the nodes
  in `(k = ln K/F, w = σ²T)`. Wings: **asymptotically flat in vol** with a C1
  exponential decay of the edge slope (a *hard* flat extrapolation is C0 and
  its kink shows up as a negative-density spike at the 10Δ strikes, the
  hard `"flat"` mode is available for comparison and the arbitrage checker
  flags its kink). Malz (1997) quadratic-in-delta smile included as
  seed/sanity model.
* Across tenors: **linear in total variance at constant forward
  log-moneyness**; flat-vol extrapolation outside the pillar range.

### No-arbitrage diagnostics (report, never silent fixes)

* Butterfly: undiscounted `C(K)` monotone/convex (density ≥ 0) **and** the
  Durrleman condition `g(k) ≥ 0` on total variance (Gatheral–Jacquier 2014).
* Calendar: `w(k,T)` non-decreasing in `T` at constant `k`.
* Violations are reported with tenor / strike / k / magnitude.

## Architecture & extension points

```
fxvol/
  conventions.py   PairConventions + registry (premium-adj flags cited)
  market_data.py   typed quote schema, CSV/JSON loaders, QuoteSource protocol
  pricing.py       Garman-Kohlhagen + full Greeks (forward form)
  delta.py         4 delta conventions, delta<->strike, PA branch logic, ATM
  smile/base.py    SmileModel ABC  <-- THE SEAM
  smile/malz.py    Malz quadratic (seed/sanity)
  smile/interpolated.py  calibrated PCHIP smile (primary)
  smile/sabr.py    Project-3 stub (interface notes)
  calibration.py   market-strangle calibration + quote round-trip
  surface.py       VolSurface: term structure, query API
  arbitrage.py     butterfly/Durrleman/calendar checks
  viz.py           3D surface, smiles, term structures, density, g(k)
  io_report.py     human-readable calibration/arbitrage report
```

**The `SmileModel` seam.** `VolSurface`, the arbitrage checks and the viz
layer touch only the `SmileModel` interface (`vol(strike)`, `nodes`,
`total_variance`, `vol_from_delta`, `price`). Swap smiles via
`VolSurface.from_market(market, smile_factory=...)`:

* **Project 2 (vanna-volga)** consumes the exposed ATM/25Δ instruments: node
  strikes/vols from `TenorCalibration`, and analytic vega/vanna/volga from
  `fxvol.pricing.greeks` 
* **Project 3 (SABR)** implements `SABRSmile(SmileModel)` (stub with plan in
  `smile/sabr.py`) fitted to the same calibrated nodes; nothing else changes.

Data sources are pluggable behind the tiny `QuoteSource` protocol
(`load() -> FxMarketData`) a Bloomberg/Refinitiv adapter didn't touch the
math.

## Sample data

`examples/eurusd_quotes.csv|json` — illustrative, realistically-shaped
EURUSD quotes (calm regime, **not live data**), vols/rates in percent, with
negative 25Δ RR (EUR downside skew). `examples/run_demo.py` runs the full
pipeline; `examples/demo.ipynb` is the guided walkthrough.

## Testing

`pytest` + `hypothesis`; ~94% line coverage. Highlights:

* **Round-trip / reprice-inputs** at every pillar for an unadjusted pair
  (EURUSD) *and* a premium-adjusted one (USDJPY-style), incl. a 2Y pillar
  exercising the forward-delta regime.
* delta↔strike round trips for all four conventions incl. deep wings;
  the PA non-monotonicity/branch test; DNS delta-neutrality for all
  conventions; the DNS sign flip.
* Greeks vs central finite differences (vanna/volga specifically).
* Market-strangle repricing, RR constraint exactness, smile-BF ≠ quoted-BF
  when RR ≠ 0, naive algebra recovered when RR = 0.
* Deliberately arb-violating surfaces (vol frown; collapsing term variance)
  are detected and located; the clean sample passes.
* Property-based monotonicity/bounds tests on pricing.

## References

* Reiswich, D. & Wystup, U. (2010). *A Guide to FX Options Quoting
  Conventions.* J. Derivatives 18(1). delta/ATM conventions, market
  strangle.
* Reiswich, D. & Wystup, U. (2012). *FX Volatility Smile Construction.*
  Wilmott. — calibration procedure.
* Clark, I. J. (2011). *Foreign Exchange Option Pricing: A Practitioner's
  Guide.* Wiley. — conventions, smile/surface construction.
* Gatheral, J. (2006). *The Volatility Surface.* Wiley; Gatheral, J. &
  Jacquier, A. (2014). *Arbitrage-free SVI volatility surfaces.* total
  variance framing, g(k) condition.
* Malz, A. (1997). *Estimating the probability distribution of the future
  exchange rate from option prices.* quadratic smile seed.
* Hagan, P. et al. (2002). *Managing Smile Risk.* SABR (Project 3).
