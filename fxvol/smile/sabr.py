"""SABR smile -- Project 3 stub (interface notes only, deliberately not
implemented in Project 1).

Plan (Hagan et al. 2002, "Managing Smile Risk"):

* ``SABRSmile(SmileModel)`` with parameters ``(alpha, beta, rho, nu)`` per
  tenor; ``vol(strike)`` = Hagan lognormal expansion around the forward.
* ``calibrate`` will consume exactly the same 5 calibrated nodes (or the raw
  ATM/RR/BF quotes plus the market-strangle machinery in
  :mod:`fxvol.calibration`) that :class:`InterpolatedSmile` uses today --
  the :class:`fxvol.smile.base.SmileModel` interface is the seam: nothing in
  :class:`fxvol.surface.VolSurface`, the arbitrage checks or the viz layer
  may depend on the concrete smile class.
* Typical FX practice: fix ``beta`` (config, default 1.0 for lognormal FX),
  fit ``(alpha, rho, nu)`` to {atm, 25d, 10d} nodes; keep the ATM constraint
  exact by solving alpha from the ATM vol cubic.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import SmileModel, SmileNode

ArrayLike = float | np.ndarray


class SABRSmile(SmileModel):
    """Placeholder for Project 3.  Instantiating raises NotImplementedError."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "SABRSmile is a Project-3 extension point; use InterpolatedSmile "
            "or MalzQuadraticSmile. See module docstring for the plan."
        )

    def vol(self, strike: ArrayLike) -> ArrayLike:  # pragma: no cover
        raise NotImplementedError

    @property
    def nodes(self) -> Sequence[SmileNode]:  # pragma: no cover
        raise NotImplementedError
