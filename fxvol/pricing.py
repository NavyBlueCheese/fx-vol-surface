"""Garman-Kohlhagen pricing and Greeks for FX vanillas (forward form).

Conventions (see :mod:`fxvol.conventions` module docstring):

* Spot ``S`` = domestic (quote) ccy per 1 unit foreign (base) ccy.
* Prices are in *domestic currency per 1 unit of foreign notional* -- i.e.
  "domestic pips" as a decimal (multiply by 10^4 for pips proper in a 4-dp
  pair).  Use :func:`convert_premium` for the other quotation styles.
* ``r_dom``/``r_for`` are continuously-compounded domestic/foreign rates.
* The forward form ``V = phi * df_dom * (F N(phi d1) - K N(phi d2))`` with
  ``F = S exp((r_dom - r_for) T)`` is used internally throughout
  [Clark2011 ch. 2].

All functions are vectorised over ``strike`` and ``vol`` via numpy
broadcasting; scalars in -> float out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr  # fast vectorised standard normal CDF

from .conventions import OptionType

ArrayLike = float | np.ndarray

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def norm_pdf(x: ArrayLike) -> np.ndarray:
    """Standard normal density."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def norm_cdf(x: ArrayLike) -> np.ndarray:
    """Standard normal CDF (scipy ``ndtr``)."""
    return ndtr(np.asarray(x, dtype=float))


def _maybe_float(x: np.ndarray) -> ArrayLike:
    """Return a python float for 0-d results, ndarray otherwise."""
    if np.ndim(x) == 0:
        return float(x)
    return x


def d1_d2(
    forward: float, strike: ArrayLike, vol: ArrayLike, expiry: float
) -> tuple[np.ndarray, np.ndarray]:
    """Black d1/d2 in forward terms.

    ``d1 = (ln(F/K) + vol^2 T / 2) / (vol sqrt(T))``, ``d2 = d1 - vol sqrt(T)``.
    Requires ``vol > 0`` and ``expiry > 0``.
    """
    strike = np.asarray(strike, dtype=float)
    vol = np.asarray(vol, dtype=float)
    if expiry <= 0.0:
        raise ValueError(f"expiry must be positive, got {expiry}")
    if np.any(vol <= 0.0):
        raise ValueError("vol must be positive")
    if np.any(strike <= 0.0):
        raise ValueError("strike must be positive")
    sqrt_t = np.sqrt(expiry)
    sig_sqrt_t = vol * sqrt_t
    d1 = (np.log(forward / strike) + 0.5 * vol * vol * expiry) / sig_sqrt_t
    d2 = d1 - sig_sqrt_t
    return d1, d2


def forward_price(
    forward: float,
    strike: ArrayLike,
    vol: ArrayLike,
    expiry: float,
    df_dom: float,
    option_type: OptionType,
) -> ArrayLike:
    """Garman-Kohlhagen price, forward form.

    Price in domestic ccy per unit foreign notional:
    ``V = phi * df_dom * (F N(phi d1) - K N(phi d2))``.
    """
    d1, d2 = d1_d2(forward, strike, vol, expiry)
    phi = option_type.phi
    strike = np.asarray(strike, dtype=float)
    v = phi * df_dom * (forward * norm_cdf(phi * d1) - strike * norm_cdf(phi * d2))
    return _maybe_float(v)


def price(
    spot: float,
    strike: ArrayLike,
    vol: ArrayLike,
    expiry: float,
    r_dom: float,
    r_for: float,
    option_type: OptionType,
) -> ArrayLike:
    """Garman-Kohlhagen price from spot and the two rates.

    ``F = S exp((r_dom - r_for) T)``; delegates to :func:`forward_price`.
    Price in domestic ccy per unit foreign notional.
    """
    fwd = spot * np.exp((r_dom - r_for) * expiry)
    df_dom = np.exp(-r_dom * expiry)
    return forward_price(fwd, strike, vol, expiry, df_dom, option_type)


@dataclass(frozen=True)
class Greeks:
    """Full Greek set for one FX vanilla.

    All values are in domestic ccy per unit foreign notional (price-space
    Greeks); ``delta_spot``/``delta_forward`` are the *unadjusted* hedge
    ratios in foreign notional units.  Premium-adjusted deltas live in
    :mod:`fxvol.delta` because they are a quoting convention, not a price
    sensitivity.

    * ``vega``  : dV/dvol         (per 1.00 of vol, i.e. per 100 vol pts)
    * ``vanna`` : d2V/dSpot dvol
    * ``volga`` : d2V/dvol2
    * ``theta`` : dV/dt (calendar decay, per year; negative of dV/dT)
    * ``rho_dom``/``rho_for``: dV/dr_dom, dV/dr_for
    """

    price: float
    delta_spot: float
    delta_forward: float
    gamma: float
    vega: float
    vanna: float
    volga: float
    theta: float
    rho_dom: float
    rho_for: float


def greeks(
    spot: float,
    strike: float,
    vol: float,
    expiry: float,
    r_dom: float,
    r_for: float,
    option_type: OptionType,
) -> Greeks:
    """Analytic Garman-Kohlhagen Greeks (scalar).

    Formulas in forward form; validated against central finite differences in
    the test suite (incl. vanna/volga, which Project 2 vanna-volga depends
    on).
    """
    fwd = spot * np.exp((r_dom - r_for) * expiry)
    df_dom = float(np.exp(-r_dom * expiry))
    df_for = float(np.exp(-r_for * expiry))
    d1_arr, d2_arr = d1_d2(fwd, strike, vol, expiry)
    d1 = float(d1_arr)
    d2 = float(d2_arr)
    phi = option_type.phi
    sqrt_t = float(np.sqrt(expiry))
    n_d1 = float(norm_pdf(d1))
    cdf_phi_d1 = float(norm_cdf(phi * d1))
    cdf_phi_d2 = float(norm_cdf(phi * d2))

    px = phi * df_dom * (fwd * cdf_phi_d1 - strike * cdf_phi_d2)
    delta_spot = phi * df_for * cdf_phi_d1
    delta_forward = phi * cdf_phi_d1
    gamma = df_for * n_d1 / (spot * vol * sqrt_t)
    vega = spot * df_for * n_d1 * sqrt_t  # = df_dom * F * n(d1) * sqrt(T)
    vanna = -df_for * n_d1 * d2 / vol
    volga = vega * d1 * d2 / vol
    theta = (
        -spot * df_for * n_d1 * vol / (2.0 * sqrt_t)
        + phi * (r_for * spot * df_for * cdf_phi_d1 - r_dom * strike * df_dom * cdf_phi_d2)
    )
    rho_dom = phi * strike * expiry * df_dom * cdf_phi_d2
    rho_for = -phi * spot * expiry * df_for * cdf_phi_d1

    return Greeks(
        price=float(px),
        delta_spot=float(delta_spot),
        delta_forward=float(delta_forward),
        gamma=float(gamma),
        vega=float(vega),
        vanna=float(vanna),
        volga=float(volga),
        theta=float(theta),
        rho_dom=float(rho_dom),
        rho_for=float(rho_for),
    )


def convert_premium(
    value_dom_per_for: ArrayLike,
    spot: float,
    strike: ArrayLike,
    style: str = "dom_per_for",
) -> ArrayLike:
    """Convert the native premium (domestic per unit foreign notional) to the
    four standard FX quotation styles [Clark2011 sec. 2.9]:

    * ``"dom_per_for"``  -- native: domestic ccy per 1 foreign notional
      ("domestic pips" as a decimal).
    * ``"pct_dom"``      -- % of domestic notional: V / K.
    * ``"for_per_for"``  -- foreign ccy per 1 foreign notional (= % of
      foreign notional): V / S.
    * ``"for_per_dom"``  -- foreign ccy per 1 domestic notional
      ("foreign pips"): V / (S K).
    """
    v = np.asarray(value_dom_per_for, dtype=float)
    k = np.asarray(strike, dtype=float)
    if style == "dom_per_for":
        out = v
    elif style == "pct_dom":
        out = v / k
    elif style == "for_per_for":
        out = v / spot
    elif style == "for_per_dom":
        out = v / (spot * k)
    else:
        raise ValueError(f"unknown premium style {style!r}")
    return _maybe_float(out)
