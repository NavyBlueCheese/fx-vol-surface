"""M5 tests: arbitrage diagnostics -- clean surface passes, deliberately
broken surfaces are flagged (with locations)."""

from __future__ import annotations

from datetime import date

from fxvol import (
    DeltaConvention,
    DeltaStyle,
    FxMarketData,
    PairConventions,
    TenorQuote,
    VolSurface,
    check_surface,
)
from fxvol.arbitrage import check_butterfly, check_calendar
from fxvol.smile import InterpolatedSmile, SmileNode
from fxvol.surface import SurfacePillar


def test_clean_sample_surface_passes(eurusd_surface: VolSurface) -> None:
    report = check_surface(eurusd_surface)
    assert report.ok, report.summary()
    assert "PASS" in report.summary()


def test_clean_pa_surface_passes(usdjpy_surface: VolSurface) -> None:
    report = check_surface(usdjpy_surface)
    assert report.ok, report.summary()


def _single_pillar_surface(vols: list[float]) -> VolSurface:
    """Hand-built one-tenor surface with chosen node vols (EURUSD-ish)."""
    fwd, t = 1.09, 0.5
    conv = DeltaConvention(DeltaStyle.SPOT, premium_adjusted=False)
    strikes = [1.02, 1.06, 1.09, 1.12, 1.16]
    nodes = [
        SmileNode(lbl, k, v)
        for lbl, k, v in zip(["10dp", "25dp", "atm", "25dc", "10dc"], strikes, vols, strict=True)
    ]
    smile = InterpolatedSmile(nodes, t, fwd, 0.98, 0.99, conv)
    pillar = SurfacePillar("6M", t, fwd, 0.98, 0.99, smile)
    return VolSurface(1.085, [pillar], PairConventions("EURUSD", False))


def test_butterfly_violation_detected() -> None:
    """A vol *frown* (concave smile) has negative wing density -> must be
    flagged by the convexity/Durrleman checks with located violations."""
    surface = _single_pillar_surface([0.055, 0.075, 0.085, 0.075, 0.055])
    violations, _ = check_butterfly(surface)
    assert violations, "arb-violating frown not detected"
    kinds = {v.kind for v in violations}
    assert kinds & {"butterfly.convexity", "butterfly.durrleman"}
    v0 = violations[0]
    assert v0.tenor == "6M" and v0.strike > 0.0  # located


def test_calendar_violation_detected() -> None:
    """Total variance dropping between tenors must be flagged."""
    conv = PairConventions("EURUSD", premium_adjusted=False)
    quotes = (
        TenorQuote("1W", atm_vol=0.14, rr_25=-0.002, bf_25=0.002,
                   r_dom=0.043, r_for=0.025),
        TenorQuote("1M", atm_vol=0.05, rr_25=-0.002, bf_25=0.002,
                   r_dom=0.043, r_for=0.025),
    )
    market = FxMarketData("EURUSD", date(2026, 6, 30), 1.085, quotes, conv)
    surface = VolSurface.from_market(market)
    violations = check_calendar(surface)
    assert violations, "calendar arbitrage not detected"
    assert all(v.kind == "calendar" for v in violations)
    assert violations[0].tenor == "1M"


def test_report_summary_lists_failures() -> None:
    surface = _single_pillar_surface([0.055, 0.075, 0.085, 0.075, 0.055])
    report = check_surface(surface)
    assert not report.ok
    s = report.summary()
    assert "FAIL" in s and "butterfly" in s
