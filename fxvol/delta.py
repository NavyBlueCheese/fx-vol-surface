"""Delta conventions and delta <-> strike conversion.

Implements all four FX delta conventions [RW2010 sec. 2; Clark2011 sec. 3.3]:

===================  ==========================================
convention           call delta
===================  ==========================================
forward, unadjusted  ``N(d1)``
spot, unadjusted     ``e^{-r_for T} N(d1)``
forward, prem-adj    ``(K/F) N(d2)``
spot, prem-adj       ``e^{-r_for T} (K/F) N(d2)``
===================  ==========================================

(put deltas: replace ``N(x)`` with ``-N(-x)``).

Premium-adjusted *call* delta is NOT monotonic in strike: it rises to a
maximum and then falls.  The market quotes correspond to the *higher-strike*
branch, where delta is decreasing in strike [RW2010 sec. 2.2.2; Clark2011
sec. 3.3.2].  :func:`strike_from_delta` locates the turning point explicitly
and root-finds on that branch; asking for a delta above the attainable
maximum raises ``ValueError`` rather than silently returning a wrong strike.

Sign conventions: call deltas are positive, put deltas negative; a quoted
"25-delta put" is ``delta = -0.25``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from scipy.optimize import brentq
from scipy.special import ndtri

from .conventions import (
    DEFAULT_NUMERICS,
    AtmConvention,
    DeltaConvention,
    DeltaStyle,
    NumericalConfig,
    OptionType,
)
from .pricing import d1_d2, norm_cdf, norm_pdf


def forward_delta(
    strike: float,
    vol: float,
    expiry: float,
    forward: float,
    option_type: OptionType,
    premium_adjusted: bool,
) -> float:
    """Forward delta (unadjusted or premium-adjusted) of a vanilla.

    Unadjusted: ``phi N(phi d1)``.  Premium-adjusted: ``phi (K/F) N(phi d2)``
    [RW2010 eq. (10)-(13)].
    """
    d1, d2 = d1_d2(forward, strike, vol, expiry)
    phi = option_type.phi
    if premium_adjusted:
        return phi * (strike / forward) * float(norm_cdf(phi * float(d2)))
    return phi * float(norm_cdf(phi * float(d1)))


def delta(
    strike: float,
    vol: float,
    expiry: float,
    forward: float,
    df_for: float,
    option_type: OptionType,
    convention: DeltaConvention,
) -> float:
    """Delta in the given convention.

    Spot delta = ``df_for x`` forward delta (both adjusted and unadjusted)
    [RW2010]; ``df_for = e^{-r_for T}``.
    """
    fd = forward_delta(
        strike, vol, expiry, forward, option_type, convention.premium_adjusted
    )
    if convention.style is DeltaStyle.SPOT:
        return df_for * fd
    return fd


def _strike_from_unadjusted_forward_delta(
    fwd_delta: float,
    vol: float,
    expiry: float,
    forward: float,
    option_type: OptionType,
) -> float:
    """Closed-form inversion for unadjusted forward delta.

    ``d1 = phi Ninv(phi delta)``; ``K = F exp(-d1 vol sqrt(T) + vol^2 T / 2)``.
    """
    phi = option_type.phi
    x = phi * fwd_delta
    if not 0.0 < x < 1.0:
        raise ValueError(
            f"unattainable {option_type.value} forward delta {fwd_delta}: "
            "phi*delta must lie in (0, 1)"
        )
    d1 = phi * float(ndtri(x))
    return forward * math.exp(-d1 * vol * math.sqrt(expiry) + 0.5 * vol * vol * expiry)


def pa_call_delta_max_strike(
    vol: float, expiry: float, forward: float
) -> tuple[float, float]:
    """Turning point of the premium-adjusted call delta.

    ``d/dK [(K/F) N(d2)] = (N(d2) - n(d2)/(vol sqrt(T))) / F``, so the
    maximum sits where ``vol sqrt(T) N(d2) = n(d2)``.  We solve that 1-D
    equation in d2 (single root: LHS-RHS is negative for very negative d2 and
    positive for large d2) and map back to the strike via
    ``K = F exp(-(d2 vol sqrt(T) + vol^2 T / 2))``.

    Returns ``(K_max, delta_max)`` where ``delta_max`` is the maximum
    attainable premium-adjusted *forward* call delta.
    """
    sig_sqrt_t = vol * math.sqrt(expiry)

    def h(d2: float) -> float:
        return sig_sqrt_t * float(norm_cdf(d2)) - float(norm_pdf(d2))

    # bracket: h > 0 for large d2 (-> sig_sqrt_t); scan downwards for h < 0.
    hi = 6.0
    while h(hi) <= 0.0:  # pragma: no cover - only for huge vols
        hi += 2.0
    lo = hi - 1.0
    while h(lo) > 0.0:
        lo -= 1.0
        if lo < -60.0:  # pragma: no cover - defensive
            raise RuntimeError("failed to bracket premium-adjusted delta turning point")
    d2_star = float(brentq(h, lo, hi, xtol=1e-14, rtol=8.9e-16))
    k_max = forward * math.exp(-(d2_star * sig_sqrt_t + 0.5 * vol * vol * expiry))
    delta_max = (k_max / forward) * float(norm_cdf(d2_star))
    return k_max, delta_max


def _strike_from_pa_forward_delta(
    fwd_delta: float,
    vol: float,
    expiry: float,
    forward: float,
    option_type: OptionType,
    numerics: NumericalConfig,
) -> float:
    """Invert premium-adjusted forward delta -> strike by root finding.

    Calls: the delta is non-monotonic; we restrict the search to the
    higher-strike branch ``K >= K_max`` where delta is decreasing -- this is
    the branch the market convention refers to [RW2010 sec. 2.2.2].
    Puts: premium-adjusted put delta is strictly decreasing in strike, so a
    simple bracketed root find suffices.
    """

    def pa_delta(k: float) -> float:
        return forward_delta(k, vol, expiry, forward, option_type, True)

    if option_type is OptionType.CALL:
        if fwd_delta <= 0.0:
            raise ValueError(f"call delta must be positive, got {fwd_delta}")
        k_max, delta_max = pa_call_delta_max_strike(vol, expiry, forward)
        if fwd_delta > delta_max + 1e-12:
            raise ValueError(
                f"premium-adjusted call delta {fwd_delta:.6f} exceeds the "
                f"maximum attainable {delta_max:.6f} for vol={vol}, T={expiry}"
            )
        if fwd_delta >= delta_max:  # numerically at the peak
            return k_max
        # expand upper bound until delta drops below target
        k_hi = k_max * 1.5
        for _ in range(200):
            if pa_delta(k_hi) < fwd_delta:
                break
            k_hi *= 1.5
        else:  # pragma: no cover - defensive
            raise RuntimeError("failed to bracket premium-adjusted call strike")
        return float(
            brentq(
                lambda k: pa_delta(k) - fwd_delta,
                k_max,
                k_hi,
                xtol=numerics.root_xtol,
                rtol=numerics.root_rtol,
                maxiter=numerics.max_iterations,
            )
        )

    # put: monotone decreasing from 0- (K -> 0) downwards
    if fwd_delta >= 0.0:
        raise ValueError(f"put delta must be negative, got {fwd_delta}")
    k_lo = forward * 1e-6
    k_hi = forward
    for _ in range(200):
        if pa_delta(k_hi) < fwd_delta:
            break
        k_hi *= 1.5
    else:  # pragma: no cover - defensive
        raise RuntimeError("failed to bracket premium-adjusted put strike")
    return float(
        brentq(
            lambda k: pa_delta(k) - fwd_delta,
            k_lo,
            k_hi,
            xtol=numerics.root_xtol,
            rtol=numerics.root_rtol,
            maxiter=numerics.max_iterations,
        )
    )


def strike_from_delta(
    delta_value: float,
    vol: float,
    expiry: float,
    forward: float,
    df_for: float,
    option_type: OptionType,
    convention: DeltaConvention,
    numerics: NumericalConfig = DEFAULT_NUMERICS,
) -> float:
    """Strike for a quoted delta under the given convention.

    Spot deltas are first rescaled to forward deltas (``/ df_for``); the
    unadjusted case then inverts in closed form, the premium-adjusted case
    root-finds on the correct branch (see module docstring).
    """
    fwd_delta = delta_value
    if convention.style is DeltaStyle.SPOT:
        fwd_delta = delta_value / df_for
    if convention.premium_adjusted:
        return _strike_from_pa_forward_delta(
            fwd_delta, vol, expiry, forward, option_type, numerics
        )
    return _strike_from_unadjusted_forward_delta(
        fwd_delta, vol, expiry, forward, option_type
    )


def atm_strike(
    atm_vol: float,
    expiry: float,
    forward: float,
    spot: float,
    atm_convention: AtmConvention,
    delta_convention: DeltaConvention,
) -> float:
    """ATM strike under the configured ATM convention.

    Delta-neutral straddle (DNS) [RW2010 sec. 3.2]:

    * unadjusted deltas:      ``K_atm = F exp(+ vol^2 T / 2)``
      (straddle delta ``N(d1) - N(-d1) = 0  =>  d1 = 0``)
    * premium-adjusted:       ``K_atm = F exp(- vol^2 T / 2)``
      (straddle delta ``(K/F)(N(d2) - N(-d2)) = 0  =>  d2 = 0`` -- note the
      sign flip vs the unadjusted case)

    The DNS strike is the same for spot and forward delta styles (the
    ``df_for`` factor scales both legs equally).
    """
    if atm_convention is AtmConvention.FORWARD:
        return forward
    if atm_convention is AtmConvention.SPOT:
        return spot
    half_var = 0.5 * atm_vol * atm_vol * expiry
    if delta_convention.premium_adjusted:
        return forward * math.exp(-half_var)
    return forward * math.exp(half_var)


def solve_smile_strike_for_delta(
    delta_value: float,
    option_type: OptionType,
    vol_of_strike: Callable[[float], float],
    expiry: float,
    forward: float,
    df_for: float,
    convention: DeltaConvention,
    numerics: NumericalConfig = DEFAULT_NUMERICS,
    vol_guess: float | None = None,
) -> tuple[float, float]:
    """Solve for the strike whose *smile* vol reproduces the target delta.

    Fixed-point iteration: ``K_{n+1} = strike_from_delta(delta, vol(K_n))``.
    This is the standard way to locate e.g. "the 25-delta call strike on the
    smile", where the vol used in the delta is itself a function of the
    strike.  Converges rapidly for realistic smiles; raises ``RuntimeError``
    on non-convergence (never silently returns a bad strike).

    Returns ``(strike, vol_at_strike)``.
    """
    vol = vol_guess if vol_guess is not None else vol_of_strike(forward)
    strike = strike_from_delta(
        delta_value, vol, expiry, forward, df_for, option_type, convention, numerics
    )
    for _ in range(numerics.fixed_point_max_iter):
        vol_new = vol_of_strike(strike)
        strike_new = strike_from_delta(
            delta_value, vol_new, expiry, forward, df_for, option_type, convention, numerics
        )
        if abs(strike_new - strike) <= numerics.fixed_point_tol * forward and abs(
            vol_new - vol
        ) <= numerics.fixed_point_tol:
            achieved = delta(
                strike_new, vol_new, expiry, forward, df_for, option_type, convention
            )
            if abs(achieved - delta_value) > 1e-8:
                raise RuntimeError(
                    f"smile strike solve converged to delta {achieved:.8f} != "
                    f"target {delta_value:.8f}"
                )
            return strike_new, vol_new
        strike, vol = strike_new, vol_new
    raise RuntimeError(
        f"smile strike fixed-point failed to converge for delta={delta_value}, "
        f"{option_type.value}, T={expiry}"
    )
