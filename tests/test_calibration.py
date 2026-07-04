"""M3 tests: market-strangle butterfly calibration (Reiswich-Wystup)."""

from __future__ import annotations

import pytest

from fxvol import (
    CalibrationError,
    FxMarketData,
    OptionType,
    calibrate_tenor,
    calibrate_wing,
    forward_price,
)
from fxvol.smile import MalzQuadraticSmile


def test_wing_reprices_market_strangle(eurusd_market: FxMarketData) -> None:
    """Acceptance: the calibrated smile wing reprices the market strangle
    price to tight tolerance, and satisfies the RR constraint exactly."""
    for pillar in eurusd_market.pillars():
        conv = eurusd_market.conventions.delta_convention(pillar.expiry)
        q = pillar.quote
        w = calibrate_wing(pillar, conv, q.atm_vol, q.rr_25, q.bf_25, 0.25)
        # (a) RR constraint
        assert w.sigma_call - w.sigma_put == pytest.approx(q.rr_25, abs=1e-14)
        # (b) strangle reprice: smile strangle price == market strangle price
        v_smile = float(
            forward_price(pillar.forward, w.strike_call, w.sigma_call,
                          pillar.expiry, pillar.df_dom, OptionType.CALL)
        ) + float(
            forward_price(pillar.forward, w.strike_put, w.sigma_put,
                          pillar.expiry, pillar.df_dom, OptionType.PUT)
        )
        assert v_smile == pytest.approx(w.market_strangle_price, abs=1e-9)
        assert abs(w.reprice_error) < 1e-9


def test_smile_strangle_vol_differs_from_market_strangle(
    eurusd_market: FxMarketData,
) -> None:
    """With rr != 0 the smile-strangle butterfly must differ from the quoted
    (market-strangle) butterfly -- if they came out equal, the calibration
    would be the naive (wrong) plug-in."""
    pillar = eurusd_market.pillars()[-1]  # 1Y: biggest |rr|
    conv = eurusd_market.conventions.delta_convention(pillar.expiry)
    q = pillar.quote
    w = calibrate_wing(pillar, conv, q.atm_vol, q.rr_25, q.bf_25, 0.25)
    assert w.smile_bf != pytest.approx(q.bf_25, abs=1e-7)
    # and the naive algebra is still a good seed: within a few bp
    naive_call = q.atm_vol + q.bf_25 + 0.5 * q.rr_25
    assert w.sigma_call == pytest.approx(naive_call, abs=5e-4)


def test_zero_rr_recovers_naive_algebra(eurusd_market: FxMarketData) -> None:
    """With rr = 0 the market and smile strangles coincide by symmetry:
    sigma_c = sigma_p = atm + bf exactly."""
    pillar = eurusd_market.pillars()[2]
    conv = eurusd_market.conventions.delta_convention(pillar.expiry)
    atm, bf = 0.082, 0.0023
    w = calibrate_wing(pillar, conv, atm, 0.0, bf, 0.25)
    assert w.sigma_call == pytest.approx(atm + bf, abs=1e-10)
    assert w.sigma_put == pytest.approx(atm + bf, abs=1e-10)


def test_five_node_smile(eurusd_market: FxMarketData) -> None:
    for pillar in eurusd_market.pillars():
        calib = calibrate_tenor(pillar, eurusd_market.conventions)
        labels = [n.label for n in calib.nodes]
        assert labels == ["10dp", "25dp", "atm", "25dc", "10dc"]
        strikes = [n.strike for n in calib.nodes]
        assert strikes == sorted(strikes)
        # ATM node carries the quoted ATM vol exactly
        atm_node = calib.nodes[2]
        assert atm_node.vol == pytest.approx(pillar.quote.atm_vol)


def test_pa_pair_calibration(usdjpy_market: FxMarketData) -> None:
    """Premium-adjusted pair calibrates and satisfies both constraints."""
    for pillar in usdjpy_market.pillars():
        calib = calibrate_tenor(pillar, usdjpy_market.conventions)
        for w in calib.wings:
            assert w.sigma_call - w.sigma_put == pytest.approx(w.rr, abs=1e-14)
            assert abs(w.reprice_error) < 1e-9
        # PA DNS strike below forward (sign flip)
        assert calib.atm_strike < pillar.forward


def test_forward_delta_convention_beyond_cutoff(usdjpy_market: FxMarketData) -> None:
    """2Y pillar is beyond the 1Y spot-delta cutoff -> forward delta."""
    from fxvol import DeltaStyle

    p2y = usdjpy_market.pillars()[-1]
    assert p2y.expiry > 1.0
    conv = usdjpy_market.conventions.delta_convention(p2y.expiry)
    assert conv.style is DeltaStyle.FORWARD
    p1m = usdjpy_market.pillars()[0]
    assert usdjpy_market.conventions.delta_convention(p1m.expiry).style is DeltaStyle.SPOT


def test_nonsense_quotes_raise(eurusd_market: FxMarketData) -> None:
    pillar = eurusd_market.pillars()[0]
    conv = eurusd_market.conventions.delta_convention(pillar.expiry)
    with pytest.raises(CalibrationError):
        calibrate_wing(pillar, conv, atm_vol=0.08, rr=0.0, bf=-0.09, delta_level=0.25)


def test_malz_seed_matches_naive_vols() -> None:
    """Malz quadratic hits atm and the naive 25d call/put vols at deltas
    0.5 / 0.25 / 0.75."""
    atm, rr, bf = 0.089, -0.0045, 0.0028
    m = MalzQuadraticSmile(atm, rr, bf, expiry=1.0, forward=1.09)
    assert m.vol_from_call_delta(0.50) == pytest.approx(atm)
    assert m.vol_from_call_delta(0.25) == pytest.approx(atm + bf + 0.5 * rr)
    assert m.vol_from_call_delta(0.75) == pytest.approx(atm + bf - 0.5 * rr)
    # vol(strike) fixed point is consistent: recompute delta from the vol
    import math

    from fxvol.pricing import d1_d2, norm_cdf

    k = 1.12
    v = float(m.vol(k))
    d1, _ = d1_d2(1.09, k, v, 1.0)
    assert m.vol_from_call_delta(float(norm_cdf(d1))) == pytest.approx(v, abs=1e-12)
    assert math.isfinite(v)
