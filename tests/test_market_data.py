"""IO tests: loaders, tenor arithmetic, rate/forward closure."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from fxvol import FxMarketData, TenorQuote, load_csv, load_json
from fxvol.market_data import FileQuoteSource, add_tenor, sort_tenors

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_csv_and_json_agree() -> None:
    m_csv = load_csv(EXAMPLES / "eurusd_quotes.csv")
    m_json = load_json(EXAMPLES / "eurusd_quotes.json")
    assert m_csv.pair == m_json.pair == "EURUSD"
    assert m_csv.spot == m_json.spot
    assert len(m_csv.quotes) == len(m_json.quotes)
    for a, b in zip(m_csv.quotes, m_json.quotes, strict=True):
        assert a == b


def test_percent_conversion() -> None:
    m = load_csv(EXAMPLES / "eurusd_quotes.csv")
    q1w = m.quotes[0]
    assert q1w.atm_vol == pytest.approx(0.076)
    assert q1w.rr_25 == pytest.approx(-0.0015)
    assert q1w.r_dom == pytest.approx(0.043)


def test_registry_fallback_conventions() -> None:
    m = load_csv(EXAMPLES / "eurusd_quotes.csv")
    assert m.conventions.pair == "EURUSD"
    assert m.conventions.premium_adjusted is False


def test_quote_source_protocol() -> None:
    src = FileQuoteSource(EXAMPLES / "eurusd_quotes.json")
    m = src.load()
    assert isinstance(m, FxMarketData)


def test_add_tenor() -> None:
    d0 = date(2026, 6, 30)
    assert add_tenor(d0, "ON") == date(2026, 7, 1)
    assert add_tenor(d0, "1W") == date(2026, 7, 7)
    assert add_tenor(d0, "1M") == date(2026, 7, 30)
    assert add_tenor(d0, "18M") == date(2027, 12, 30)
    assert add_tenor(d0, "1Y") == date(2027, 6, 30)
    # end-of-month clamp
    assert add_tenor(date(2026, 1, 31), "1M") == date(2026, 2, 28)
    with pytest.raises(ValueError):
        add_tenor(d0, "1X")


def test_sort_tenors() -> None:
    d0 = date(2026, 6, 30)
    assert sort_tenors(["1Y", "1W", "3M", "ON"], d0) == ["ON", "1W", "3M", "1Y"]


def test_forward_implies_foreign_rate() -> None:
    """Given forward + r_dom, r_for is implied from covered interest parity."""
    import math

    spot, rd, t_label = 1.0850, 0.043, "6M"
    conv_market = load_csv(EXAMPLES / "eurusd_quotes.csv")
    # target: same pillar as the rates-based market
    base = next(p for p in conv_market.pillars() if p.tenor == t_label)
    q = TenorQuote(t_label, atm_vol=0.085, rr_25=-0.003, bf_25=0.002,
                   r_dom=rd, forward=base.forward)
    m = FxMarketData("EURUSD", conv_market.valuation_date, spot, (q,),
                     conv_market.conventions)
    p = m.pillars()[0]
    assert p.r_for == pytest.approx(rd - math.log(base.forward / spot) / p.expiry)
    assert p.forward == pytest.approx(base.forward)


def test_inconsistent_forward_and_rate_raises() -> None:
    m0 = load_csv(EXAMPLES / "eurusd_quotes.csv")
    q = TenorQuote("6M", atm_vol=0.085, rr_25=-0.003, bf_25=0.002,
                   r_dom=0.043, r_for=0.010, forward=1.10)
    m = FxMarketData("EURUSD", m0.valuation_date, 1.085, (q,), m0.conventions)
    with pytest.raises(ValueError, match="inconsistent"):
        m.pillars()


def test_missing_rate_raises() -> None:
    m0 = load_csv(EXAMPLES / "eurusd_quotes.csv")
    q = TenorQuote("6M", atm_vol=0.085, rr_25=-0.003, bf_25=0.002)
    m = FxMarketData("EURUSD", m0.valuation_date, 1.085, (q,), m0.conventions)
    with pytest.raises(ValueError):
        m.pillars()


def test_lonely_10d_quote_rejected() -> None:
    with pytest.raises(ValueError, match="together"):
        TenorQuote("6M", atm_vol=0.085, rr_25=-0.003, bf_25=0.002, rr_10=-0.005)


def test_unknown_pair_needs_explicit_conventions() -> None:
    from fxvol import get_pair_conventions

    with pytest.raises(KeyError, match="premium-adjustment"):
        get_pair_conventions("SEKNOK")
