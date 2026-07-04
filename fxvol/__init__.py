"""fxvol -- production-grade FX implied volatility surface engine.

Quotes (ATM / risk-reversal / butterfly in delta space) -> calibrated,
arbitrage-checked implied vol surface with full FX convention handling.

See README for the math and the SmileModel extension seam (vanna-volga /
SABR attach points).
"""

from .arbitrage import ArbitrageReport, check_surface
from .calibration import (
    CalibrationError,
    ImpliedQuotes,
    TenorCalibration,
    WingCalibration,
    calibrate_market,
    calibrate_tenor,
    calibrate_wing,
    implied_quotes_from_smile,
)
from .conventions import (
    DEFAULT_NUMERICS,
    PAIR_REGISTRY,
    AtmConvention,
    DayCount,
    DeltaConvention,
    DeltaStyle,
    NumericalConfig,
    OptionType,
    PairConventions,
    get_pair_conventions,
)
from .delta import atm_strike, delta, forward_delta, strike_from_delta
from .market_data import (
    FileQuoteSource,
    FxMarketData,
    MarketPillar,
    TenorQuote,
    load_csv,
    load_json,
)
from .pricing import Greeks, convert_premium, forward_price, greeks, price
from .smile import InterpolatedSmile, MalzQuadraticSmile, SABRSmile, SmileModel, SmileNode
from .surface import SurfacePillar, VolSurface

__version__ = "0.1.0"

__all__ = [
    "ArbitrageReport",
    "AtmConvention",
    "CalibrationError",
    "DayCount",
    "DEFAULT_NUMERICS",
    "DeltaConvention",
    "DeltaStyle",
    "FileQuoteSource",
    "FxMarketData",
    "Greeks",
    "ImpliedQuotes",
    "InterpolatedSmile",
    "MalzQuadraticSmile",
    "MarketPillar",
    "NumericalConfig",
    "OptionType",
    "PAIR_REGISTRY",
    "PairConventions",
    "SABRSmile",
    "SmileModel",
    "SmileNode",
    "SurfacePillar",
    "TenorCalibration",
    "TenorQuote",
    "VolSurface",
    "WingCalibration",
    "atm_strike",
    "calibrate_market",
    "calibrate_tenor",
    "calibrate_wing",
    "check_surface",
    "convert_premium",
    "delta",
    "forward_delta",
    "forward_price",
    "get_pair_conventions",
    "greeks",
    "implied_quotes_from_smile",
    "load_csv",
    "load_json",
    "price",
    "strike_from_delta",
]
