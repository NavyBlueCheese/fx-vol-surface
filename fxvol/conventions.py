"""FX market conventions: delta styles, premium adjustment, ATM definitions.

Conventions are first-class configuration in this package.  Every quoted FX
volatility number is meaningless until you fix:

* the *delta style*  -- spot delta vs forward delta,
* whether the delta is *premium-adjusted* (premium paid in foreign/base ccy),
* the *ATM convention* -- delta-neutral straddle (DNS), ATM-forward, ATM-spot,
* the day count used to turn tenors into year fractions.

References
----------
- Reiswich, D. & Wystup, U. (2010), "A Guide to FX Options Quoting
  Conventions", The Journal of Derivatives 18(1).  [RW2010]
- Reiswich, D. & Wystup, U. (2012), "FX Volatility Smile Construction",
  Wilmott.  [RW2012]
- Clark, I. J. (2011), "Foreign Exchange Option Pricing: A Practitioner's
  Guide", Wiley, ch. 3.  [Clark2011]

Units convention used throughout the package
--------------------------------------------
FX rate S = units of *domestic* (quote) currency per 1 unit of *foreign*
(base) currency.  E.g. for EURUSD, EUR is foreign/base, USD is domestic/quote,
S = USD per EUR.  All option prices are in domestic currency per 1 unit of
foreign notional unless stated otherwise.  Vols and rates are decimals
(0.08 = 8%), times are year fractions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class OptionType(str, Enum):
    """Vanilla option type. ``phi`` is the usual +1/-1 sign."""

    CALL = "call"
    PUT = "put"

    @property
    def phi(self) -> int:
        return 1 if self is OptionType.CALL else -1


class DeltaStyle(str, Enum):
    """Spot delta (hedge in spot) vs forward delta (hedge in forward)."""

    SPOT = "spot"
    FORWARD = "forward"


class AtmConvention(str, Enum):
    """ATM strike definition.

    DELTA_NEUTRAL (DNS) is the interbank standard for most pairs and tenors
    [RW2010 sec. 3.2]; ATM_FORWARD and ATM_SPOT are provided for completeness.
    """

    DELTA_NEUTRAL = "dns"
    FORWARD = "atmf"
    SPOT = "spot"


class DayCount(str, Enum):
    """Day count for converting tenor dates to year fractions.

    ACT365F (act/365 fixed) is the package default for vol time; configurable
    per pair.
    """

    ACT365F = "act/365f"
    ACT360 = "act/360"

    def year_fraction(self, days: int) -> float:
        if self is DayCount.ACT365F:
            return days / 365.0
        return days / 360.0


@dataclass(frozen=True)
class DeltaConvention:
    """A fully-specified delta convention: style x premium adjustment."""

    style: DeltaStyle
    premium_adjusted: bool

    def describe(self) -> str:
        pa = "premium-adjusted" if self.premium_adjusted else "unadjusted"
        return f"{self.style.value} delta, {pa}"


@dataclass(frozen=True)
class NumericalConfig:
    """Numerical tolerances used by solvers.  Tolerances are configuration,
    not literals buried in code."""

    root_xtol: float = 1e-14
    root_rtol: float = 8.9e-16  # ~4*eps, scipy brentq minimum
    max_iterations: int = 200
    fixed_point_tol: float = 1e-12
    fixed_point_max_iter: int = 100
    #: acceptance tolerance for repricing calibrated instruments (price units,
    #: domestic ccy per unit foreign notional)
    reprice_price_tol: float = 1e-9
    #: acceptance tolerance on vols (1e-6 = 0.0001 vol pt, far inside the
    #: 0.01 vol pt target)
    reprice_vol_tol: float = 1e-6


DEFAULT_NUMERICS = NumericalConfig()


@dataclass(frozen=True)
class PairConventions:
    """Market conventions for one currency pair.

    Parameters
    ----------
    pair:
        6-letter pair, e.g. ``"EURUSD"`` (foreign+domestic).
    premium_adjusted:
        True when the option premium is conventionally paid in the *foreign*
        (base) currency, in which case quoted deltas are premium-adjusted
        [RW2010 sec. 2.2].  E.g. USDJPY premia are paid in USD (the base) so
        USDJPY deltas are premium-adjusted; EURUSD premia are paid in USD
        (the quote/domestic ccy) so EURUSD deltas are unadjusted.
    spot_delta_cutoff_years:
        Deltas are *spot* deltas for expiries with ``T <= cutoff`` and
        *forward* deltas beyond.  Market practice for most pairs is spot
        delta up to and including 1Y and forward delta for longer tenors
        [Clark2011 sec. 3.3]; the cutoff is itself a convention, hence
        configurable.
    atm_convention:
        ATM strike definition, default delta-neutral straddle [RW2010].
    day_count:
        Day count for vol time, default act/365 fixed.
    """

    pair: str
    premium_adjusted: bool
    spot_delta_cutoff_years: float = 1.0
    atm_convention: AtmConvention = AtmConvention.DELTA_NEUTRAL
    day_count: DayCount = DayCount.ACT365F

    def delta_convention(self, expiry: float) -> DeltaConvention:
        """Delta convention applying at year fraction ``expiry``.

        Spot delta for ``expiry <= spot_delta_cutoff_years`` (inclusive,
        matching the common "up to and including 1Y" phrasing), forward
        delta beyond.
        """
        style = (
            DeltaStyle.SPOT
            if expiry <= self.spot_delta_cutoff_years + 1e-12
            else DeltaStyle.FORWARD
        )
        return DeltaConvention(style=style, premium_adjusted=self.premium_adjusted)

    @property
    def foreign_ccy(self) -> str:
        return self.pair[:3]

    @property
    def domestic_ccy(self) -> str:
        return self.pair[3:]

    def with_(self, **changes: object) -> PairConventions:
        """Return a copy with fields replaced (convenience for experiments)."""
        return replace(self, **changes)  # type: ignore[arg-type]


#: Registry of example pairs with documented defaults.
#:
#: Premium-adjustment flags follow RW2010 (Table 1/sec. 2.2) and Clark2011
#: (sec. 3.3): deltas are premium-adjusted when the premium currency is the
#: base (foreign) currency of the pair.  In the interbank market the premium
#: currency is USD for USD pairs, EUR for EUR crosses without USD, else the
#: more "major" currency of the pair.
#:
#: These are *defaults* -- override per trade/desk via ``PairConventions``.
PAIR_REGISTRY: dict[str, PairConventions] = {
    # USD is the domestic/quote ccy and the premium ccy -> unadjusted.
    "EURUSD": PairConventions("EURUSD", premium_adjusted=False),
    "GBPUSD": PairConventions("GBPUSD", premium_adjusted=False),
    "AUDUSD": PairConventions("AUDUSD", premium_adjusted=False),
    "NZDUSD": PairConventions("NZDUSD", premium_adjusted=False),
    # USD is the foreign/base ccy and the premium ccy -> premium-adjusted.
    "USDJPY": PairConventions("USDJPY", premium_adjusted=True),
    "USDCHF": PairConventions("USDCHF", premium_adjusted=True),
    "USDCAD": PairConventions("USDCAD", premium_adjusted=True),
    "USDBRL": PairConventions("USDBRL", premium_adjusted=True),
    # EUR crosses: premium in EUR (the base ccy) -> premium-adjusted.
    "EURJPY": PairConventions("EURJPY", premium_adjusted=True),
    "EURGBP": PairConventions("EURGBP", premium_adjusted=True),
}


def get_pair_conventions(pair: str) -> PairConventions:
    """Look up registry defaults for ``pair`` (case-insensitive).

    Raises ``KeyError`` with a helpful message for unknown pairs: unknown
    pairs must be configured explicitly rather than silently guessed.
    """
    key = pair.upper().replace("/", "")
    try:
        return PAIR_REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"No registered conventions for pair {pair!r}. Construct a "
            "PairConventions explicitly (the premium-adjustment flag is a "
            "market convention you must supply; see Reiswich-Wystup 2010)."
        ) from None
