from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from fxvol import (
    FxMarketData,
    PairConventions,
    TenorQuote,
    VolSurface,
    load_csv,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="session")
def eurusd_market() -> FxMarketData:
    return load_csv(EXAMPLES / "eurusd_quotes.csv")


@pytest.fixture(scope="session")
def eurusd_surface(eurusd_market: FxMarketData) -> VolSurface:
    return VolSurface.from_market(eurusd_market)


@pytest.fixture(scope="session")
def usdjpy_market() -> FxMarketData:
    """Synthetic premium-adjusted pair: exercises the PA machinery
    end-to-end.  Vols/rates as decimals here (constructed, not loaded)."""
    quotes = (
        TenorQuote("1M", atm_vol=0.098, rr_25=-0.012, bf_25=0.0022,
                   rr_10=-0.022, bf_10=0.0080, r_dom=0.005, r_for=0.043),
        TenorQuote("3M", atm_vol=0.101, rr_25=-0.015, bf_25=0.0026,
                   rr_10=-0.028, bf_10=0.0092, r_dom=0.005, r_for=0.043),
        TenorQuote("1Y", atm_vol=0.106, rr_25=-0.019, bf_25=0.0031,
                   rr_10=-0.036, bf_10=0.0110, r_dom=0.005, r_for=0.043),
        TenorQuote("2Y", atm_vol=0.108, rr_25=-0.021, bf_25=0.0034,
                   rr_10=-0.040, bf_10=0.0120, r_dom=0.005, r_for=0.043),
    )
    return FxMarketData(
        pair="USDJPY",
        valuation_date=date(2026, 6, 30),
        spot=155.00,
        quotes=quotes,
        conventions=PairConventions("USDJPY", premium_adjusted=True),
    )


@pytest.fixture(scope="session")
def usdjpy_surface(usdjpy_market: FxMarketData) -> VolSurface:
    return VolSurface.from_market(usdjpy_market)
