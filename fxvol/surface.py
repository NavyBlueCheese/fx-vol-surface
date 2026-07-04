"""VolSurface: per-tenor smiles + term-structure interpolation.

Term structure: total implied variance ``w(k, T) = vol(k, T)^2 T`` is
interpolated **linearly in T at constant forward log-moneyness**
``k = ln(K / F(T))`` between pillar tenors [Gatheral 2006 ch. 4; Clark2011
sec. 4.5].  Variance is additive in time, so linear-in-w interpolation is the
standard choice and keeps calendar arbitrage controllable (checked, not
assumed, by :mod:`fxvol.arbitrage`).

Extrapolation in T: flat in *vol* at constant k before the first pillar and
beyond the last (i.e. ``w`` scales proportionally with T) -- documented
choice, guaranteed calendar-arbitrage-free in the extrapolated regions.

Curves: pillar zero rates are interpolated linearly in T (flat beyond the
ends); forwards then follow from covered interest parity, so the forward
curve is consistent with the discount factors used in pricing at any T.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .calibration import (
    TenorCalibration,
    build_smile,
    calibrate_market,
)
from .conventions import (
    DEFAULT_NUMERICS,
    DeltaConvention,
    NumericalConfig,
    OptionType,
    PairConventions,
)
from .delta import strike_from_delta
from .market_data import FxMarketData
from .pricing import Greeks, forward_price
from .pricing import greeks as gk_greeks
from .smile import SmileModel

ArrayLike = float | np.ndarray


def _interp_flat(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    """Piecewise-linear with flat extrapolation."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x) - 1
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] * (1.0 - t) + ys[i + 1] * t


@dataclass(frozen=True)
class SurfacePillar:
    """One tenor of the surface: market context + its smile."""

    tenor: str
    expiry: float
    forward: float
    df_dom: float
    df_for: float
    smile: SmileModel
    calibration: TenorCalibration | None = None


class VolSurface:
    """Arbitrage-checked FX implied vol surface.

    Query API (all vols decimals, times year fractions):

    * :meth:`vol` -- implied vol at (strike, T)
    * :meth:`vol_from_delta` -- implied vol + strike at (quoted delta, T)
    * :meth:`total_variance` -- w(k, T)
    * :meth:`price`, :meth:`greeks` -- Garman-Kohlhagen off the surface
    * :meth:`forward`, :meth:`df_dom`, :meth:`df_for` -- curves
    """

    def __init__(
        self,
        spot: float,
        pillars: Sequence[SurfacePillar],
        conventions: PairConventions,
        numerics: NumericalConfig = DEFAULT_NUMERICS,
    ) -> None:
        if not pillars:
            raise ValueError("need at least one pillar")
        self.spot = spot
        self.pillars = sorted(pillars, key=lambda p: p.expiry)
        self.conventions = conventions
        self.numerics = numerics
        self._ts = [p.expiry for p in self.pillars]
        # pillar zero rates from dfs: r = -ln(df)/T
        self._r_dom = [-math.log(p.df_dom) / p.expiry for p in self.pillars]
        self._r_for = [-math.log(p.df_for) / p.expiry for p in self.pillars]

    # -- construction --------------------------------------------------------
    @classmethod
    def from_market(
        cls,
        market: FxMarketData,
        smile_factory: Callable[[TenorCalibration], SmileModel] | None = None,
        numerics: NumericalConfig = DEFAULT_NUMERICS,
    ) -> VolSurface:
        """Calibrate all tenors (market-strangle procedure) and assemble the
        surface.  ``smile_factory`` is the SmileModel seam: pass a factory
        producing any SmileModel (e.g. a SABR fit in Project 3)."""
        calibs = calibrate_market(market, numerics)
        pillars = [
            SurfacePillar(
                tenor=c.pillar.tenor,
                expiry=c.pillar.expiry,
                forward=c.pillar.forward,
                df_dom=c.pillar.df_dom,
                df_for=c.pillar.df_for,
                smile=build_smile(c, smile_factory),  # type: ignore[arg-type]
                calibration=c,
            )
            for c in calibs
        ]
        return cls(market.spot, pillars, market.conventions, numerics)

    # -- curves ---------------------------------------------------------------
    def zero_rate_dom(self, expiry: float) -> float:
        """Domestic zero rate, linear in T, flat extrapolation."""
        return _interp_flat(expiry, self._ts, self._r_dom)

    def zero_rate_for(self, expiry: float) -> float:
        """Foreign zero rate, linear in T, flat extrapolation."""
        return _interp_flat(expiry, self._ts, self._r_for)

    def df_dom(self, expiry: float) -> float:
        return math.exp(-self.zero_rate_dom(expiry) * expiry)

    def df_for(self, expiry: float) -> float:
        return math.exp(-self.zero_rate_for(expiry) * expiry)

    def forward(self, expiry: float) -> float:
        """Outright forward from covered interest parity off the zero curves."""
        return self.spot * math.exp(
            (self.zero_rate_dom(expiry) - self.zero_rate_for(expiry)) * expiry
        )

    # -- vol queries ----------------------------------------------------------
    def _pillar_w(self, i: int, k: ArrayLike) -> np.ndarray:
        """Total variance of pillar i at forward log-moneyness k."""
        p = self.pillars[i]
        strikes = p.forward * np.exp(np.asarray(k, dtype=float))
        v = np.asarray(p.smile.vol(strikes), dtype=float)
        return v * v * p.expiry

    def total_variance(self, log_moneyness: ArrayLike, expiry: float) -> ArrayLike:
        """w(k, T), linear in T at constant k between pillars, flat-vol
        extrapolation outside the pillar range (w proportional to T)."""
        if expiry <= 0.0:
            raise ValueError("expiry must be positive")
        k = np.asarray(log_moneyness, dtype=float)
        scalar = np.ndim(k) == 0
        k = np.atleast_1d(k)
        ts = self._ts
        if expiry <= ts[0]:
            w = self._pillar_w(0, k) * (expiry / ts[0])
        elif expiry >= ts[-1]:
            w = self._pillar_w(len(ts) - 1, k) * (expiry / ts[-1])
        else:
            i = bisect.bisect_right(ts, expiry) - 1
            if abs(ts[i] - expiry) < 1e-14:
                w = self._pillar_w(i, k)
            else:
                t0, t1 = ts[i], ts[i + 1]
                lam = (expiry - t0) / (t1 - t0)
                w = (1.0 - lam) * self._pillar_w(i, k) + lam * self._pillar_w(i + 1, k)
        return float(w[0]) if scalar else w

    def vol(self, strike: ArrayLike, expiry: float) -> ArrayLike:
        """Implied vol at (strike, T)."""
        fwd = self.forward(expiry)
        k = np.log(np.asarray(strike, dtype=float) / fwd)
        w = np.asarray(self.total_variance(k, expiry), dtype=float)
        v = np.sqrt(w / expiry)
        return float(v) if np.ndim(v) == 0 else v

    def vol_from_delta(
        self, delta_value: float, expiry: float, option_type: OptionType
    ) -> tuple[float, float]:
        """(strike, vol) at a quoted delta and arbitrary T.

        The delta convention follows the pair's tenor rule (spot/forward
        cutoff) at this T.  Fixed-point iteration on (strike, vol).
        """
        conv: DeltaConvention = self.conventions.delta_convention(expiry)
        fwd = self.forward(expiry)
        dff = self.df_for(expiry)
        vol = float(np.asarray(self.vol(fwd, expiry), dtype=float))
        strike = strike_from_delta(
            delta_value, vol, expiry, fwd, dff, option_type, conv, self.numerics
        )
        for _ in range(self.numerics.fixed_point_max_iter):
            vol_new = float(np.asarray(self.vol(strike, expiry), dtype=float))
            strike_new = strike_from_delta(
                delta_value, vol_new, expiry, fwd, dff, option_type, conv, self.numerics
            )
            if (
                abs(strike_new - strike) <= self.numerics.fixed_point_tol * fwd
                and abs(vol_new - vol) <= self.numerics.fixed_point_tol
            ):
                return strike_new, vol_new
            strike, vol = strike_new, vol_new
        raise RuntimeError(
            f"vol_from_delta failed to converge: delta={delta_value}, T={expiry}"
        )

    # -- pricing off the surface ----------------------------------------------
    def price(self, strike: ArrayLike, expiry: float, option_type: OptionType) -> ArrayLike:
        """Vanilla price off the surface (domestic ccy / unit foreign)."""
        return forward_price(
            self.forward(expiry),
            strike,
            self.vol(strike, expiry),
            expiry,
            self.df_dom(expiry),
            option_type,
        )

    def greeks(self, strike: float, expiry: float, option_type: OptionType) -> Greeks:
        """Full analytic Greeks at the surface vol (sticky-strike)."""
        vol = float(np.asarray(self.vol(strike, expiry), dtype=float))
        return gk_greeks(
            self.spot,
            strike,
            vol,
            expiry,
            self.zero_rate_dom(expiry),
            self.zero_rate_for(expiry),
            option_type,
        )

    # -- introspection ----------------------------------------------------------
    def pillar(self, tenor: str) -> SurfacePillar:
        for p in self.pillars:
            if p.tenor.upper() == tenor.upper():
                return p
        raise KeyError(f"no pillar {tenor!r}; have {[p.tenor for p in self.pillars]}")

    @property
    def expiries(self) -> list[float]:
        return list(self._ts)
