"""Calibrated 5-node interpolated smile (primary Project-1 model).

Interpolation: shape-preserving monotone cubic (PCHIP) through the calibrated
nodes in ``(k, w)`` space, with ``k = ln(K/F)`` forward log-moneyness and
``w = vol^2 T`` total variance.  PCHIP avoids the overshoot/ringing of a
natural cubic spline through only 5 points.

Extrapolation beyond the outermost (10-delta) nodes -- documented choice,
because *wing extrapolation is where arbitrage sneaks in*:

* ``"smooth_flat"`` (default): asymptotically flat in vol, with the edge
  slope ``dvol/dk`` decayed exponentially over a band of scale
  ``extrap_decay`` (default: half the node span).  This is C1 at the last
  node, so the implied density stays bounded; a *hard* flat extrapolation is
  C0 only and its kink shows up as a negative density spike at the 10d
  strikes (a real butterfly violation, not a numerical artifact).
* ``"flat"``: hard flat-in-vol beyond the wings.  Simple and bounded but
  kinked at the last node -- the arbitrage checker will flag the kink.
  Provided for comparison/teaching.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..conventions import DeltaConvention
from .base import SmileModel, SmileNode

ArrayLike = float | np.ndarray


class InterpolatedSmile(SmileModel):
    """PCHIP total-variance smile through calibrated nodes.

    Parameters
    ----------
    nodes:
        Calibrated smile nodes; strikes must be strictly increasing and vols
        positive.  Typically 5 nodes {10dp, 25dp, atm, 25dc, 10dc}, but any
        >= 3 monotone-strike set works (e.g. 3 nodes when only 25d quotes
        exist).
    extrapolation:
        ``"smooth_flat"`` (default) or ``"flat"`` -- see module docstring.
    extrap_decay:
        Decay scale (in k units) of the smooth wing transition; default
        half the node k-span.
    """

    def __init__(
        self,
        nodes: Sequence[SmileNode],
        expiry: float,
        forward: float,
        df_dom: float,
        df_for: float,
        delta_convention: DeltaConvention,
        extrapolation: str = "smooth_flat",
        extrap_decay: float | None = None,
    ) -> None:
        if len(nodes) < 3:
            raise ValueError("need at least 3 smile nodes")
        if extrapolation not in ("smooth_flat", "flat"):
            raise ValueError(f"unknown extrapolation mode {extrapolation!r}")
        strikes = np.array([n.strike for n in nodes], dtype=float)
        vols = np.array([n.vol for n in nodes], dtype=float)
        if np.any(np.diff(strikes) <= 0.0):
            raise ValueError(
                "smile node strikes must be strictly increasing, got "
                f"{strikes.tolist()} — check quotes/conventions"
            )
        if np.any(vols <= 0.0):
            raise ValueError("smile node vols must be positive")
        self.expiry = expiry
        self.forward = forward
        self.df_dom = df_dom
        self.df_for = df_for
        self.delta_convention = delta_convention
        self.extrapolation = extrapolation
        self._nodes = tuple(nodes)
        self._k = np.log(strikes / forward)
        self._w = vols * vols * expiry
        self._pchip = PchipInterpolator(self._k, self._w, extrapolate=False)
        self._dw = self._pchip.derivative()
        self._k_lo = float(self._k[0])
        self._k_hi = float(self._k[-1])
        self._vol_lo = float(vols[0])
        self._vol_hi = float(vols[-1])
        # edge slopes dvol/dk = w'/(2 vol T)
        self._slope_lo = float(self._dw(self._k_lo)) / (2.0 * self._vol_lo * expiry)
        self._slope_hi = float(self._dw(self._k_hi)) / (2.0 * self._vol_hi * expiry)
        span = self._k_hi - self._k_lo
        self._decay = extrap_decay if extrap_decay is not None else 0.5 * span
        if self._decay <= 0.0:
            raise ValueError("extrap_decay must be positive")

    @property
    def nodes(self) -> Sequence[SmileNode]:
        return self._nodes

    def _extrapolate(self, k: np.ndarray, upper: bool) -> np.ndarray:
        """Wing vols beyond the outermost node (see module docstring)."""
        if upper:
            edge_vol, slope, h = self._vol_hi, self._slope_hi, k - self._k_hi
        else:
            edge_vol, slope, h = self._vol_lo, -self._slope_lo, self._k_lo - k
        if self.extrapolation == "flat":
            return np.full(k.shape, edge_vol)
        # C1 smooth-flat: vol' decays as slope*exp(-h/tau); vol -> edge + slope*tau
        tau = self._decay
        return edge_vol + slope * tau * (1.0 - np.exp(-h / tau))

    def vol(self, strike: ArrayLike) -> ArrayLike:
        k = np.log(np.asarray(strike, dtype=float) / self.forward)
        scalar = np.ndim(k) == 0
        k = np.atleast_1d(np.asarray(k, dtype=float))
        out = np.empty(k.shape, dtype=float)
        inside = (k >= self._k_lo) & (k <= self._k_hi)
        below = k < self._k_lo
        above = k > self._k_hi
        if np.any(inside):
            w = self._pchip(k[inside])
            out[inside] = np.sqrt(np.maximum(w, 0.0) / self.expiry)
        if np.any(below):
            out[below] = self._extrapolate(k[below], upper=False)
        if np.any(above):
            out[above] = self._extrapolate(k[above], upper=True)
        return float(out[0]) if scalar else out

    def asymptotic_wing_vols(self) -> tuple[float, float]:
        """(put-wing, call-wing) limiting vols of the smooth-flat wings."""
        if self.extrapolation == "flat":
            return self._vol_lo, self._vol_hi
        return (
            self._vol_lo - self._slope_lo * self._decay,
            self._vol_hi + self._slope_hi * self._decay,
        )
