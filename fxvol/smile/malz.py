"""Malz (1997) quadratic-in-delta smile.

Closed-form seed / sanity model:

    ``vol(delta_c) = atm - 2 rr (delta_c - 1/2) + 16 bf (delta_c - 1/2)^2``

where ``delta_c`` is the *unadjusted forward call delta* ``N(d1)``
[Malz 1997].  At ``delta_c = 0.25`` this gives ``atm + bf + rr/2`` (the naive
25d call vol) and at ``delta_c = 0.75`` (the 25d put) ``atm + bf - rr/2`` --
i.e. Malz treats the quoted butterfly as a *smile* strangle, which is exactly
the simplification the market-strangle calibration in
:mod:`fxvol.calibration` corrects.  Use Malz for seeds and sanity checks, not
as the production smile.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..conventions import DeltaConvention, DeltaStyle
from ..pricing import d1_d2, norm_cdf
from .base import SmileModel, SmileNode

ArrayLike = float | np.ndarray

#: Malz is defined in unadjusted forward call-delta space (module docstring)
_UNADJUSTED_FORWARD_DELTA = DeltaConvention(DeltaStyle.FORWARD, premium_adjusted=False)


class MalzQuadraticSmile(SmileModel):
    """Quadratic smile in unadjusted forward call delta.

    Note: the model is *defined* in unadjusted forward-delta space; the
    ``delta_convention`` attribute is carried only so the generic
    ``vol_from_delta`` API works, and defaults to unadjusted forward delta.
    """

    def __init__(
        self,
        atm_vol: float,
        rr_25: float,
        bf_25: float,
        expiry: float,
        forward: float,
        df_dom: float = 1.0,
        df_for: float = 1.0,
        delta_convention: DeltaConvention = _UNADJUSTED_FORWARD_DELTA,
    ) -> None:
        self.atm_vol = atm_vol
        self.rr_25 = rr_25
        self.bf_25 = bf_25
        self.expiry = expiry
        self.forward = forward
        self.df_dom = df_dom
        self.df_for = df_for
        self.delta_convention = delta_convention

    def vol_from_call_delta(self, call_delta: ArrayLike) -> ArrayLike:
        """Malz quadratic in unadjusted forward call delta (vectorised)."""
        d = np.asarray(call_delta, dtype=float)
        x = d - 0.5
        v = self.atm_vol - 2.0 * self.rr_25 * x + 16.0 * self.bf_25 * x * x
        return float(v) if np.ndim(v) == 0 else v

    def vol(self, strike: ArrayLike) -> ArrayLike:
        """Vol at strike via fixed-point on delta.

        ``delta = N(d1(K, vol(delta)))`` -- converges in a handful of
        iterations for realistic (atm, rr, bf).
        """
        k = np.asarray(strike, dtype=float)
        scalar = np.ndim(k) == 0
        k = np.atleast_1d(k)
        vol = np.full(k.shape, self.atm_vol)
        for _ in range(100):
            d1, _ = d1_d2(self.forward, k, vol, self.expiry)
            dlt = norm_cdf(d1)
            vol_new = np.asarray(self.vol_from_call_delta(dlt), dtype=float)
            if np.max(np.abs(vol_new - vol)) < 1e-14:
                vol = vol_new
                break
            vol = vol_new
        else:  # pragma: no cover - defensive
            raise RuntimeError("Malz vol(strike) fixed point failed to converge")
        return float(vol[0]) if scalar else vol

    @property
    def nodes(self) -> Sequence[SmileNode]:
        """The three defining points (in call-delta space)."""
        return (
            SmileNode("25dp", np.nan, float(self.vol_from_call_delta(0.75))),
            SmileNode("atm", np.nan, float(self.vol_from_call_delta(0.5))),
            SmileNode("25dc", np.nan, float(self.vol_from_call_delta(0.25))),
        )
