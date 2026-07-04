"""M1 tests: put-call parity, known values, Greeks vs finite differences."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fxvol import OptionType, convert_premium, forward_price, greeks, price
from fxvol.pricing import d1_d2, norm_cdf

S, RD, RF = 1.0850, 0.0430, 0.0250


def _fwd(t: float) -> float:
    return S * math.exp((RD - RF) * t)


@pytest.mark.parametrize("k_mult", [0.85, 0.95, 1.0, 1.05, 1.20])
@pytest.mark.parametrize("t", [0.02, 0.25, 1.0, 2.0])
@pytest.mark.parametrize("vol", [0.05, 0.10, 0.25])
def test_put_call_parity(k_mult: float, t: float, vol: float) -> None:
    """C - P = df_dom (F - K), forward consistency built in."""
    f = _fwd(t)
    k = f * k_mult
    c = price(S, k, vol, t, RD, RF, OptionType.CALL)
    p = price(S, k, vol, t, RD, RF, OptionType.PUT)
    assert c - p == pytest.approx(math.exp(-RD * t) * (f - k), abs=1e-14)


def test_atm_forward_closed_form() -> None:
    """K = F: straddle-half closed form C = df F (2N(vol sqrt(T)/2) - 1)."""
    t, vol = 0.75, 0.09
    f = _fwd(t)
    c = price(S, f, vol, t, RD, RF, OptionType.CALL)
    expected = math.exp(-RD * t) * f * (2.0 * float(norm_cdf(0.5 * vol * math.sqrt(t))) - 1.0)
    assert c == pytest.approx(expected, abs=1e-15)


def test_deep_itm_limits() -> None:
    t, vol = 0.5, 0.08
    f = _fwd(t)
    df = math.exp(-RD * t)
    # K -> 0: call -> df*(F - K); put -> 0
    k = 0.2 * f
    assert price(S, k, vol, t, RD, RF, OptionType.CALL) == pytest.approx(df * (f - k), rel=1e-10)
    assert price(S, k, vol, t, RD, RF, OptionType.PUT) == pytest.approx(0.0, abs=1e-12)


def test_forward_form_equals_spot_form() -> None:
    t, vol, k = 0.25, 0.082, 1.10
    f = _fwd(t)
    assert forward_price(f, k, vol, t, math.exp(-RD * t), OptionType.CALL) == pytest.approx(
        price(S, k, vol, t, RD, RF, OptionType.CALL), abs=1e-16
    )


def test_vectorised_over_strikes() -> None:
    t, vol = 0.5, 0.085
    ks = np.linspace(0.9, 1.3, 7)
    v = price(S, ks, vol, t, RD, RF, OptionType.CALL)
    assert isinstance(v, np.ndarray) and v.shape == ks.shape
    for i, k in enumerate(ks):
        assert v[i] == pytest.approx(price(S, float(k), vol, t, RD, RF, OptionType.CALL))


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        d1_d2(1.0, 1.0, 0.1, 0.0)
    with pytest.raises(ValueError):
        d1_d2(1.0, 1.0, -0.1, 1.0)
    with pytest.raises(ValueError):
        d1_d2(1.0, -1.0, 0.1, 1.0)


@pytest.mark.parametrize("opt", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("k", [0.95, 1.085, 1.09, 1.25])
@pytest.mark.parametrize("t", [0.1, 1.0])
def test_greeks_vs_finite_difference(opt: OptionType, k: float, t: float) -> None:
    """Every analytic Greek agrees with a central finite difference.

    Vanna/volga get explicit coverage -- Project 2 (vanna-volga) depends on
    them.
    """
    vol = 0.085
    g = greeks(S, k, vol, t, RD, RF, opt)

    # step sizes balance truncation vs cancellation noise (price ~0.1,
    # double precision => 2nd-diff noise ~ eps*price/h^2)
    hs = S * 1e-4
    hv = 1e-4
    ht = 1e-6
    hr = 1e-6

    def p(s=S, v=vol, tt=t, rd=RD, rf=RF) -> float:
        return float(price(s, k, v, tt, rd, rf, opt))

    delta_fd = (p(s=S + hs) - p(s=S - hs)) / (2 * hs)
    gamma_fd = (p(s=S + hs) - 2 * p() + p(s=S - hs)) / hs**2
    vega_fd = (p(v=vol + hv) - p(v=vol - hv)) / (2 * hv)
    volga_fd = (p(v=vol + hv) - 2 * p() + p(v=vol - hv)) / hv**2
    vanna_fd = (
        p(s=S + hs, v=vol + hv) - p(s=S - hs, v=vol + hv)
        - p(s=S + hs, v=vol - hv) + p(s=S - hs, v=vol - hv)
    ) / (4 * hs * hv)
    theta_fd = -(p(tt=t + ht) - p(tt=t - ht)) / (2 * ht)
    rho_d_fd = (p(rd=RD + hr) - p(rd=RD - hr)) / (2 * hr)
    rho_f_fd = (p(rf=RF + hr) - p(rf=RF - hr)) / (2 * hr)

    assert g.delta_spot == pytest.approx(delta_fd, abs=1e-7)
    assert g.gamma == pytest.approx(gamma_fd, rel=1e-3, abs=1e-6)
    assert g.vega == pytest.approx(vega_fd, rel=1e-6, abs=1e-9)
    assert g.volga == pytest.approx(volga_fd, rel=5e-3, abs=1e-6)
    assert g.vanna == pytest.approx(vanna_fd, rel=1e-3, abs=1e-6)
    assert g.theta == pytest.approx(theta_fd, rel=1e-5, abs=1e-8)
    assert g.rho_dom == pytest.approx(rho_d_fd, rel=1e-6, abs=1e-9)
    assert g.rho_for == pytest.approx(rho_f_fd, rel=1e-6, abs=1e-9)
    # forward delta consistency: delta_spot = df_for * delta_forward
    assert g.delta_spot == pytest.approx(math.exp(-RF * t) * g.delta_forward, abs=1e-14)


def test_premium_conversions() -> None:
    t, vol, k = 0.5, 0.085, 1.10
    v = float(price(S, k, vol, t, RD, RF, OptionType.CALL))
    assert convert_premium(v, S, k, "dom_per_for") == v
    assert convert_premium(v, S, k, "pct_dom") == pytest.approx(v / k)
    assert convert_premium(v, S, k, "for_per_for") == pytest.approx(v / S)
    assert convert_premium(v, S, k, "for_per_dom") == pytest.approx(v / (S * k))
    with pytest.raises(ValueError):
        convert_premium(v, S, k, "bogus")
