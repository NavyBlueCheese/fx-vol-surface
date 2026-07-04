"""Market-strangle butterfly calibration (Reiswich-Wystup) per tenor.

The quoted butterfly ``bf`` is a *market strangle* (a.k.a. broker fly)
number, not the smile convexity directly [RW2012 sec. 3; Clark2011 sec. 3.6].
The procedure implemented here, per wing delta level (25d, then 10d):

1. Market strangle vol: ``sigma_ms = atm + bf``.
2. Market strangle strikes ``K_c_ms, K_p_ms``: strikes at delta +/-level
   using the *single* vol ``sigma_ms`` for both legs, in the pair's delta
   convention.
3. Market strangle price (THE observable the bf quote encodes):
   ``v_ms = Call(K_c_ms, sigma_ms) + Put(K_p_ms, sigma_ms)``.
4. Solve for smile wing vols ``(sigma_c, sigma_p)`` s.t.

   a. ``sigma_c - sigma_p = rr``  (risk-reversal constraint), and
   b. with *smile* strikes ``K_c, K_p`` defined at delta +/-level using the
      smile's own wing vols, the smile reprices the market strangle:
      ``Call(K_c, sigma_c) + Put(K_p, sigma_p) = v_ms``.

   (a) reduces the problem to a 1-D root find in ``sigma_c``, solved with
   Brent on an expanding bracket seeded by the naive smile-strangle vols
   ``atm + bf +/- rr/2`` (excellent initial guesses).  Convergence and
   reprice accuracy are asserted -- never silently return a bad smile.

The smile-strangle vol generally differs from ``sigma_ms``; the difference
grows with |rr| and wing convexity, and vanishes when rr = 0.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from scipy.optimize import brentq

from .conventions import (
    DEFAULT_NUMERICS,
    DeltaConvention,
    NumericalConfig,
    OptionType,
    PairConventions,
)
from .delta import atm_strike, strike_from_delta
from .market_data import FxMarketData, MarketPillar
from .pricing import forward_price
from .smile import InterpolatedSmile, SmileModel, SmileNode


class CalibrationError(RuntimeError):
    """Raised when a smile calibration fails to converge or reprice."""


@dataclass(frozen=True)
class WingCalibration:
    """Diagnostics for one wing (delta level) of one tenor."""

    delta_level: float  # e.g. 0.25
    rr: float
    bf: float
    sigma_ms: float  # market strangle vol = atm + bf
    strike_call_ms: float
    strike_put_ms: float
    market_strangle_price: float
    sigma_call: float  # calibrated smile wing vols
    sigma_put: float
    strike_call: float  # smile strikes at +/- delta_level with own vols
    strike_put: float
    reprice_error: float  # smile strangle price - market strangle price
    smile_bf: float  # smile-strangle butterfly 0.5*(sc+sp) - atm (diagnostic)


@dataclass(frozen=True)
class TenorCalibration:
    """Full calibration result for one tenor pillar."""

    pillar: MarketPillar
    delta_convention: DeltaConvention
    atm_strike: float
    atm_vol: float
    nodes: tuple[SmileNode, ...]  # sorted by strike
    wings: tuple[WingCalibration, ...]


def _market_strangle(
    pillar: MarketPillar,
    conv: DeltaConvention,
    sigma_ms: float,
    delta_level: float,
    numerics: NumericalConfig,
) -> tuple[float, float, float]:
    """(K_call_ms, K_put_ms, v_ms) for a single-vol strangle at +/-delta."""
    k_c = strike_from_delta(
        +delta_level, sigma_ms, pillar.expiry, pillar.forward, pillar.df_for,
        OptionType.CALL, conv, numerics,
    )
    k_p = strike_from_delta(
        -delta_level, sigma_ms, pillar.expiry, pillar.forward, pillar.df_for,
        OptionType.PUT, conv, numerics,
    )
    v = float(
        forward_price(pillar.forward, k_c, sigma_ms, pillar.expiry, pillar.df_dom, OptionType.CALL)
    ) + float(
        forward_price(pillar.forward, k_p, sigma_ms, pillar.expiry, pillar.df_dom, OptionType.PUT)
    )
    return k_c, k_p, v


def calibrate_wing(
    pillar: MarketPillar,
    conv: DeltaConvention,
    atm_vol: float,
    rr: float,
    bf: float,
    delta_level: float,
    numerics: NumericalConfig = DEFAULT_NUMERICS,
) -> WingCalibration:
    """Solve one wing's smile vols from (atm, rr, bf) -- see module docstring."""
    sigma_ms = atm_vol + bf
    if sigma_ms <= 0.0:
        raise CalibrationError(
            f"{pillar.tenor}: market strangle vol atm+bf = {sigma_ms} <= 0"
        )
    k_c_ms, k_p_ms, v_ms = _market_strangle(pillar, conv, sigma_ms, delta_level, numerics)

    def smile_strangle_price(sigma_c: float) -> tuple[float, float, float]:
        """Price of the smile strangle for candidate sigma_c (returns
        (price, K_c, K_p)); sigma_p is tied by the RR constraint."""
        sigma_p = sigma_c - rr
        k_c = strike_from_delta(
            +delta_level, sigma_c, pillar.expiry, pillar.forward, pillar.df_for,
            OptionType.CALL, conv, numerics,
        )
        k_p = strike_from_delta(
            -delta_level, sigma_p, pillar.expiry, pillar.forward, pillar.df_for,
            OptionType.PUT, conv, numerics,
        )
        v = float(
            forward_price(pillar.forward, k_c, sigma_c, pillar.expiry, pillar.df_dom, OptionType.CALL)
        ) + float(
            forward_price(pillar.forward, k_p, sigma_p, pillar.expiry, pillar.df_dom, OptionType.PUT)
        )
        return v, k_c, k_p

    def objective(sigma_c: float) -> float:
        return smile_strangle_price(sigma_c)[0] - v_ms

    # Seed: naive smile-strangle algebra sigma_c = atm + bf + rr/2 (exact when
    # the market and smile strangles coincide, e.g. rr = 0).
    seed = atm_vol + bf + 0.5 * rr
    # Both wing vols must stay positive throughout the bracket.
    floor = max(1e-4, rr + 1e-4)  # sigma_p = sigma_c - rr > 0
    lo = hi = min(max(seed, floor + 1e-4), 5.0)
    f_seed = objective(lo)
    if abs(f_seed) < numerics.reprice_price_tol:
        sigma_c = lo
    else:
        solved: float | None = None
        width = max(0.05 * atm_vol, 0.002)
        for _ in range(30):
            lo = max(lo - width, floor)
            hi = hi + width
            f_lo, f_hi = objective(lo), objective(hi)
            if f_lo == 0.0:
                solved = lo
                break
            if f_hi == 0.0:
                solved = hi
                break
            if f_lo * f_hi < 0.0:
                solved = float(
                    brentq(
                        objective, lo, hi,
                        xtol=numerics.root_xtol, rtol=numerics.root_rtol,
                        maxiter=numerics.max_iterations,
                    )
                )
                break
            width *= 2.0
        if solved is None:
            raise CalibrationError(
                f"{pillar.tenor} {delta_level:.0%} wing: could not bracket the "
                f"smile-strangle root (atm={atm_vol}, rr={rr}, bf={bf})"
            )
        sigma_c = solved

    v_smile, k_c, k_p = smile_strangle_price(sigma_c)
    err = v_smile - v_ms
    if abs(err) > max(numerics.reprice_price_tol, 1e-12 * v_ms) * 10:
        raise CalibrationError(
            f"{pillar.tenor} {delta_level:.0%} wing: strangle reprice error "
            f"{err:.3e} exceeds tolerance"
        )
    sigma_p = sigma_c - rr
    if sigma_p <= 0.0:
        raise CalibrationError(
            f"{pillar.tenor} {delta_level:.0%} wing: negative put vol {sigma_p}"
        )
    return WingCalibration(
        delta_level=delta_level,
        rr=rr,
        bf=bf,
        sigma_ms=sigma_ms,
        strike_call_ms=k_c_ms,
        strike_put_ms=k_p_ms,
        market_strangle_price=v_ms,
        sigma_call=sigma_c,
        sigma_put=sigma_p,
        strike_call=k_c,
        strike_put=k_p,
        reprice_error=err,
        smile_bf=0.5 * (sigma_c + sigma_p) - atm_vol,
    )


def calibrate_tenor(
    pillar: MarketPillar,
    conventions: PairConventions,
    numerics: NumericalConfig = DEFAULT_NUMERICS,
) -> TenorCalibration:
    """Calibrate the 5-node (or 3-node if no 10d quotes) smile for one tenor."""
    q = pillar.quote
    conv = conventions.delta_convention(pillar.expiry)
    # S is only needed for the (rare) ATM_SPOT convention; recover it from
    # F = S e^{(r_dom - r_for)T} = S df_for/df_dom  =>  S = F df_dom/df_for.
    k_atm = atm_strike(
        q.atm_vol, pillar.expiry, pillar.forward,
        spot=pillar.forward * pillar.df_dom / pillar.df_for,
        atm_convention=conventions.atm_convention,
        delta_convention=conv,
    )
    wings: list[WingCalibration] = []
    wings.append(
        calibrate_wing(pillar, conv, q.atm_vol, q.rr_25, q.bf_25, 0.25, numerics)
    )
    if q.has_10d:
        assert q.rr_10 is not None and q.bf_10 is not None
        wings.append(
            calibrate_wing(pillar, conv, q.atm_vol, q.rr_10, q.bf_10, 0.10, numerics)
        )

    nodes: list[SmileNode] = [
        SmileNode("atm", k_atm, q.atm_vol, delta=None, option_type=None)
    ]
    for w in wings:
        lbl = f"{w.delta_level:.0%}".rstrip("%")
        nodes.append(
            SmileNode(f"{lbl}dc", w.strike_call, w.sigma_call, +w.delta_level, OptionType.CALL)
        )
        nodes.append(
            SmileNode(f"{lbl}dp", w.strike_put, w.sigma_put, -w.delta_level, OptionType.PUT)
        )
    nodes.sort(key=lambda n: n.strike)
    strikes = [n.strike for n in nodes]
    if any(b - a <= 0.0 for a, b in zip(strikes, strikes[1:], strict=False)):
        raise CalibrationError(
            f"{pillar.tenor}: calibrated node strikes not strictly increasing: "
            f"{[(n.label, round(n.strike, 6)) for n in nodes]}"
        )
    return TenorCalibration(
        pillar=pillar,
        delta_convention=conv,
        atm_strike=k_atm,
        atm_vol=q.atm_vol,
        nodes=tuple(nodes),
        wings=tuple(wings),
    )


def build_smile(
    calib: TenorCalibration,
    smile_factory: Callable[[TenorCalibration], InterpolatedSmile] | None = None,
) -> InterpolatedSmile:
    """Construct the tenor smile from a calibration (default: PCHIP
    interpolated total-variance smile).  Custom factories (e.g. a future
    SABRSmile fit) receive the full calibration."""
    if smile_factory is not None:
        return smile_factory(calib)
    p = calib.pillar
    return InterpolatedSmile(
        nodes=calib.nodes,
        expiry=p.expiry,
        forward=p.forward,
        df_dom=p.df_dom,
        df_for=p.df_for,
        delta_convention=calib.delta_convention,
    )


def calibrate_market(
    market: FxMarketData,
    numerics: NumericalConfig = DEFAULT_NUMERICS,
) -> list[TenorCalibration]:
    """Calibrate every tenor of a market snapshot (sorted by expiry)."""
    return [
        calibrate_tenor(p, market.conventions, numerics) for p in market.pillars()
    ]


# ---------------------------------------------------------------------------
# Round-trip: recover (atm, rr, bf) quotes back from a calibrated smile.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpliedQuotes:
    """Quotes implied *back* from a smile -- for the round-trip acceptance
    test: these must match the input quotes to tolerance."""

    tenor: str
    atm_vol: float
    rr_25: float
    bf_25: float
    rr_10: float | None
    bf_10: float | None


def implied_quotes_from_smile(
    smile: SmileModel,
    pillar: MarketPillar,
    conventions: PairConventions,
    delta_levels: Sequence[float] = (0.25, 0.10),
    numerics: NumericalConfig = DEFAULT_NUMERICS,
) -> ImpliedQuotes:
    """Recover (atm, rr, bf) from an arbitrary smile.

    * ATM: fixed point of ``vol(atm_strike(vol))`` under the ATM convention.
    * RR: smile vols at the +/-delta smile strikes.
    * BF: the vol premium ``b`` over recovered ATM such that a single-vol
      market strangle (vol ``atm + b``, strikes at +/-delta with that same
      vol) has the same price as the smile strangle -- i.e. exactly inverting
      the market-strangle definition.  Solved with Brent.
    """
    conv = conventions.delta_convention(pillar.expiry)

    # ATM fixed point
    vol = float(smile.vol(pillar.forward))
    for _ in range(numerics.fixed_point_max_iter):
        k_atm = atm_strike(
            vol, pillar.expiry, pillar.forward,
            spot=pillar.forward * pillar.df_dom / pillar.df_for,
            atm_convention=conventions.atm_convention,
            delta_convention=conv,
        )
        vol_new = float(smile.vol(k_atm))
        if abs(vol_new - vol) < numerics.fixed_point_tol:
            vol = vol_new
            break
        vol = vol_new
    atm_rec = vol

    def wing(delta_level: float) -> tuple[float, float]:
        k_c, s_c = smile.vol_from_delta(+delta_level, OptionType.CALL, numerics)
        k_p, s_p = smile.vol_from_delta(-delta_level, OptionType.PUT, numerics)
        rr = s_c - s_p
        v_smile = float(
            forward_price(pillar.forward, k_c, s_c, pillar.expiry, pillar.df_dom, OptionType.CALL)
        ) + float(
            forward_price(pillar.forward, k_p, s_p, pillar.expiry, pillar.df_dom, OptionType.PUT)
        )

        def obj(b: float) -> float:
            _, _, v_ms = _market_strangle(pillar, conv, atm_rec + b, delta_level, numerics)
            return v_ms - v_smile

        lo, hi = -0.5 * atm_rec, 2.0 * atm_rec
        bf = float(
            brentq(obj, lo, hi, xtol=numerics.root_xtol, rtol=numerics.root_rtol,
                   maxiter=numerics.max_iterations)
        )
        return rr, bf

    rr25, bf25 = wing(delta_levels[0])
    rr10: float | None = None
    bf10: float | None = None
    if pillar.quote.has_10d and len(delta_levels) > 1:
        rr10, bf10 = wing(delta_levels[1])
    return ImpliedQuotes(
        tenor=pillar.tenor, atm_vol=atm_rec, rr_25=rr25, bf_25=bf25,
        rr_10=rr10, bf_10=bf10,
    )
