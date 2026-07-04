"""SmileModel interface -- the seam for Projects 2 and 3.

A :class:`SmileModel` is one tenor's smile: vol as a function of strike, plus
enough market context (forward, discount factors, delta convention) to
convert between strike and delta space and to price vanillas on itself.

Any implementation (Malz quadratic, calibrated interpolant, SABR later) is
interchangeable inside :class:`fxvol.surface.VolSurface` -- the surface only
touches this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..conventions import (
    DEFAULT_NUMERICS,
    DeltaConvention,
    NumericalConfig,
    OptionType,
)
from ..delta import delta as delta_fn
from ..delta import solve_smile_strike_for_delta
from ..pricing import forward_price

ArrayLike = float | np.ndarray


@dataclass(frozen=True)
class SmileNode:
    """One calibrated smile node (e.g. the 25-delta call point)."""

    label: str  # e.g. "10dp", "25dp", "atm", "25dc", "10dc"
    strike: float
    vol: float
    delta: float | None = None  # quoted delta in the pair's convention
    option_type: OptionType | None = None


class SmileModel(ABC):
    """Abstract smile for a single tenor.

    Required context (constructor args of implementations): ``expiry`` (year
    fraction), ``forward``, ``df_dom``, ``df_for`` and the tenor's
    ``DeltaConvention``.  All vols are decimals.
    """

    expiry: float
    forward: float
    df_dom: float
    df_for: float
    delta_convention: DeltaConvention

    # -- core interface ----------------------------------------------------
    @abstractmethod
    def vol(self, strike: ArrayLike) -> ArrayLike:
        """Implied vol at ``strike`` (vectorised)."""

    @property
    @abstractmethod
    def nodes(self) -> Sequence[SmileNode]:
        """Calibrated nodes (ATM + wings) -- exposed for vanna-volga
        (Project 2), reporting and testing."""

    # -- derived conveniences ------------------------------------------------
    def total_variance(self, log_moneyness: ArrayLike) -> ArrayLike:
        """Total implied variance ``w(k) = vol(K)^2 T`` at forward
        log-moneyness ``k = ln(K/F)``."""
        k = np.asarray(log_moneyness, dtype=float)
        strikes = self.forward * np.exp(k)
        v = np.asarray(self.vol(strikes), dtype=float)
        w = v * v * self.expiry
        return float(w) if np.ndim(w) == 0 else w

    def vol_from_delta(
        self,
        delta_value: float,
        option_type: OptionType,
        numerics: NumericalConfig = DEFAULT_NUMERICS,
    ) -> tuple[float, float]:
        """(strike, vol) on this smile at a quoted delta (pair convention)."""
        return solve_smile_strike_for_delta(
            delta_value,
            option_type,
            lambda k: float(np.asarray(self.vol(k), dtype=float)),
            self.expiry,
            self.forward,
            self.df_for,
            self.delta_convention,
            numerics,
        )

    def price(self, strike: ArrayLike, option_type: OptionType) -> ArrayLike:
        """Vanilla price off the smile (domestic ccy / unit foreign)."""
        return forward_price(
            self.forward, strike, self.vol(strike), self.expiry, self.df_dom, option_type
        )

    def delta_at(self, strike: float, option_type: OptionType) -> float:
        """Quoted-convention delta at ``strike`` using the smile vol."""
        return delta_fn(
            strike,
            float(np.asarray(self.vol(strike), dtype=float)),
            self.expiry,
            self.forward,
            self.df_for,
            option_type,
            self.delta_convention,
        )
