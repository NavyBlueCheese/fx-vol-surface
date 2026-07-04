"""Property-based tests (Hypothesis): pricing monotonicities and bounds."""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from fxvol import OptionType, price

spots = st.floats(0.5, 2.0)
vols = st.floats(0.02, 0.60)
expiries = st.floats(0.02, 2.0)
rates = st.floats(-0.01, 0.08)
k_mults = st.floats(0.6, 1.8)


@settings(max_examples=200, deadline=None)
@given(s=spots, vol=vols, t=expiries, rd=rates, rf=rates, km=k_mults)
def test_call_price_bounds(s: float, vol: float, t: float, rd: float, rf: float,
                           km: float) -> None:
    """df_dom * max(F - K, 0) <= C <= df_dom * F  (undiscounted forward bounds)."""
    f = s * math.exp((rd - rf) * t)
    k = f * km
    df = math.exp(-rd * t)
    c = float(price(s, k, vol, t, rd, rf, OptionType.CALL))
    assert c >= df * max(f - k, 0.0) - 1e-12
    assert c <= df * f + 1e-12


@settings(max_examples=200, deadline=None)
@given(s=spots, vol=vols, t=expiries, rd=rates, rf=rates, km=k_mults)
def test_price_increasing_in_vol(s: float, vol: float, t: float, rd: float,
                                 rf: float, km: float) -> None:
    f = s * math.exp((rd - rf) * t)
    k = f * km
    lo = float(price(s, k, vol, t, rd, rf, OptionType.CALL))
    hi = float(price(s, k, vol + 0.02, t, rd, rf, OptionType.CALL))
    assert hi >= lo - 1e-14


@settings(max_examples=200, deadline=None)
@given(s=spots, vol=vols, t=expiries, rd=rates, rf=rates, km=k_mults)
def test_call_decreasing_put_increasing_in_strike(
    s: float, vol: float, t: float, rd: float, rf: float, km: float
) -> None:
    f = s * math.exp((rd - rf) * t)
    k = f * km
    c1 = float(price(s, k, vol, t, rd, rf, OptionType.CALL))
    c2 = float(price(s, k * 1.02, vol, t, rd, rf, OptionType.CALL))
    p1 = float(price(s, k, vol, t, rd, rf, OptionType.PUT))
    p2 = float(price(s, k * 1.02, vol, t, rd, rf, OptionType.PUT))
    assert c2 <= c1 + 1e-14
    assert p2 >= p1 - 1e-14


@settings(max_examples=200, deadline=None)
@given(s=spots, vol=vols, t=expiries, rd=rates, rf=rates, km=k_mults)
def test_parity_property(s: float, vol: float, t: float, rd: float, rf: float,
                         km: float) -> None:
    f = s * math.exp((rd - rf) * t)
    k = f * km
    c = float(price(s, k, vol, t, rd, rf, OptionType.CALL))
    p = float(price(s, k, vol, t, rd, rf, OptionType.PUT))
    assert abs(c - p - math.exp(-rd * t) * (f - k)) < 1e-10


@settings(max_examples=100, deadline=None)
@given(vol=st.floats(0.03, 0.4), t=st.floats(0.05, 2.0), km=st.floats(0.7, 1.6))
def test_strike_delta_roundtrip_property(vol: float, t: float, km: float) -> None:
    """Round trip strike->delta->strike across random smiles/conventions."""
    from fxvol import DeltaConvention, DeltaStyle, delta, strike_from_delta
    from fxvol.delta import pa_call_delta_max_strike

    f, dff = 1.09, math.exp(-0.02 * t)
    k0 = f * km
    for pa in (False, True):
        for style in (DeltaStyle.SPOT, DeltaStyle.FORWARD):
            conv = DeltaConvention(style, pa)
            opt = OptionType.PUT
            d = delta(k0, vol, t, f, dff, opt, conv)
            # deltas saturated at +/-1 (or +/-df) to machine precision are
            # not invertible -- N(d1) has lost all resolution
            scale = dff if style is DeltaStyle.SPOT else 1.0
            if abs(d) < 1e-6 or abs(d) / scale > 1.0 - 1e-9:
                continue
            assert strike_from_delta(d, vol, t, f, dff, opt, conv) == (
                __import__("pytest").approx(k0, rel=1e-8)
            )
            # calls: only the market branch is invertible for PA
            if pa:
                k_max, _ = pa_call_delta_max_strike(vol, t, f)
                if k0 < k_max:
                    continue
            d = delta(k0, vol, t, f, dff, OptionType.CALL, conv)
            if d < 1e-6 or d / scale > 1.0 - 1e-9:
                continue
            assert strike_from_delta(d, vol, t, f, dff, OptionType.CALL, conv) == (
                __import__("pytest").approx(k0, rel=1e-8)
            )
