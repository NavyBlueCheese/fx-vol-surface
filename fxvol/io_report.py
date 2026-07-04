"""Human-readable calibration + arbitrage report."""

from __future__ import annotations

from collections.abc import Sequence

from .arbitrage import ArbitrageReport
from .calibration import ImpliedQuotes, TenorCalibration, implied_quotes_from_smile
from .market_data import FxMarketData
from .surface import VolSurface


def _pct(x: float | None) -> str:
    return "     -" if x is None else f"{x * 100:6.3f}"


def calibration_report(
    market: FxMarketData,
    surface: VolSurface,
    arb: ArbitrageReport | None = None,
) -> str:
    """Full text report: conventions, per-tenor calibration, round-trip
    reprice errors and (optionally) arbitrage diagnostics."""
    conv = market.conventions
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"FX vol surface calibration report -- {market.pair}")
    lines.append(f"valuation date : {market.valuation_date}   spot: {market.spot}")
    lines.append(
        f"conventions    : premium-adjusted={conv.premium_adjusted}, "
        f"atm={conv.atm_convention.value}, spot-delta cutoff="
        f"{conv.spot_delta_cutoff_years}y, day count={conv.day_count.value}"
    )
    lines.append("=" * 78)

    calibs: Sequence[TenorCalibration] = [
        p.calibration for p in surface.pillars if p.calibration is not None
    ]
    for c in calibs:
        p = c.pillar
        lines.append(
            f"\n{p.tenor:>4}  T={p.expiry:.6f}  F={p.forward:.6f}  "
            f"df_dom={p.df_dom:.6f}  delta conv: {c.delta_convention.describe()}"
        )
        lines.append(f"      ATM strike {c.atm_strike:.6f}  vol {c.atm_vol * 100:.3f}%")
        for w in c.wings:
            lines.append(
                f"      {w.delta_level:.0%} wing: sigma_ms={w.sigma_ms * 100:.3f}%  "
                f"v_ms={w.market_strangle_price:.8f}"
            )
            lines.append(
                f"        smile: call {w.sigma_call * 100:.3f}% @K={w.strike_call:.6f}  "
                f"put {w.sigma_put * 100:.3f}% @K={w.strike_put:.6f}  "
                f"smile-bf={w.smile_bf * 100:.3f}% (quoted bf={w.bf * 100:.3f}%)  "
                f"reprice err={w.reprice_error:.2e}"
            )

    lines.append("\n" + "-" * 78)
    lines.append("Round-trip: quotes implied back from the calibrated surface")
    lines.append(
        f"{'tenor':>5} | {'ATM in':>7} {'ATM out':>7} | {'RR25 in':>7} {'RR25 out':>8} | "
        f"{'BF25 in':>7} {'BF25 out':>8} | {'RR10 in':>7} {'RR10 out':>8} | "
        f"{'BF10 in':>7} {'BF10 out':>8} | max err (vol pts)"
    )
    max_err_overall = 0.0
    for sp in surface.pillars:
        cal = sp.calibration
        if cal is None:
            continue
        q = cal.pillar.quote
        imp: ImpliedQuotes = implied_quotes_from_smile(sp.smile, cal.pillar, market.conventions)
        errs = [abs(imp.atm_vol - q.atm_vol), abs(imp.rr_25 - q.rr_25), abs(imp.bf_25 - q.bf_25)]
        if (
            q.rr_10 is not None
            and q.bf_10 is not None
            and imp.rr_10 is not None
            and imp.bf_10 is not None
        ):
            errs += [abs(imp.rr_10 - q.rr_10), abs(imp.bf_10 - q.bf_10)]
        max_err = max(errs)
        max_err_overall = max(max_err_overall, max_err)
        lines.append(
            f"{sp.tenor:>5} | {_pct(q.atm_vol)} {_pct(imp.atm_vol)} | "
            f"{_pct(q.rr_25)} {_pct(imp.rr_25):>8} | {_pct(q.bf_25)} {_pct(imp.bf_25):>8} | "
            f"{_pct(q.rr_10)} {_pct(imp.rr_10):>8} | {_pct(q.bf_10)} {_pct(imp.bf_10):>8} | "
            f"{max_err * 100:.2e}"
        )
    lines.append(f"worst round-trip error: {max_err_overall * 100:.3e} vol points")

    if arb is not None:
        lines.append("\n" + "-" * 78)
        lines.append(arb.summary())
    lines.append("=" * 78)
    return "\n".join(lines)
