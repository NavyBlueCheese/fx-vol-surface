"""M6: the human-readable calibration/arbitrage report."""

from __future__ import annotations

from fxvol import FxMarketData, VolSurface, check_surface
from fxvol.io_report import calibration_report


def test_report_contents(eurusd_market: FxMarketData, eurusd_surface: VolSurface) -> None:
    arb = check_surface(eurusd_surface)
    text = calibration_report(eurusd_market, eurusd_surface, arb)
    assert "EURUSD" in text
    assert "Round-trip" in text
    assert "PASS" in text  # clean sample set
    for tenor in ("1W", "1M", "3M", "6M", "1Y"):
        assert tenor in text
    # every tenor line shows sub-tolerance round-trip errors; the worst-case
    # line must be present and small
    worst_line = next(ln for ln in text.splitlines() if ln.startswith("worst"))
    worst = float(worst_line.split(":")[1].split("vol")[0])
    assert worst < 1e-4  # < 0.0001 vol points, far inside the 0.01 target


def test_report_without_arbitrage_section(eurusd_market: FxMarketData,
                                          eurusd_surface: VolSurface) -> None:
    text = calibration_report(eurusd_market, eurusd_surface, None)
    assert "Arbitrage" not in text
