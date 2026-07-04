"""M2 tests: delta conventions, delta<->strike round trips, premium-adjusted
branch selection, DNS ATM."""

from __future__ import annotations

import math

import pytest

from fxvol import (
    AtmConvention,
    DeltaConvention,
    DeltaStyle,
    OptionType,
    atm_strike,
    delta,
    forward_delta,
    strike_from_delta,
)
from fxvol.delta import pa_call_delta_max_strike

F, T, DF_FOR = 1.0900, 0.5, math.exp(-0.025 * 0.5)
VOL = 0.085

ALL_CONVENTIONS = [
    DeltaConvention(DeltaStyle.SPOT, premium_adjusted=False),
    DeltaConvention(DeltaStyle.FORWARD, premium_adjusted=False),
    DeltaConvention(DeltaStyle.SPOT, premium_adjusted=True),
    DeltaConvention(DeltaStyle.FORWARD, premium_adjusted=True),
]


@pytest.mark.parametrize("conv", ALL_CONVENTIONS, ids=lambda c: c.describe())
@pytest.mark.parametrize("opt", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("k_mult", [0.85, 0.95, 1.0, 1.05, 1.15, 1.30])
def test_strike_delta_strike_roundtrip(
    conv: DeltaConvention, opt: OptionType, k_mult: float
) -> None:
    """strike -> delta -> strike must return the original strike, including
    deep wings, for every convention."""
    k0 = F * k_mult
    d = delta(k0, VOL, T, F, DF_FOR, opt, conv)
    if conv.premium_adjusted and opt is OptionType.CALL:
        # only strikes on the market (higher-strike) branch are invertible
        k_max, _ = pa_call_delta_max_strike(VOL, T, F)
        if k0 < k_max:
            pytest.skip("strike below PA-call turning point: not on market branch")
    k1 = strike_from_delta(d, VOL, T, F, DF_FOR, opt, conv)
    assert k1 == pytest.approx(k0, rel=1e-10)


@pytest.mark.parametrize("conv", ALL_CONVENTIONS, ids=lambda c: c.describe())
@pytest.mark.parametrize(
    "opt,d",
    [
        (OptionType.CALL, 0.25),
        (OptionType.CALL, 0.10),
        (OptionType.CALL, 0.45),
        (OptionType.PUT, -0.25),
        (OptionType.PUT, -0.10),
        (OptionType.PUT, -0.45),
    ],
)
def test_delta_strike_delta_roundtrip(
    conv: DeltaConvention, opt: OptionType, d: float
) -> None:
    """delta -> strike -> delta round trip for every convention."""
    k = strike_from_delta(d, VOL, T, F, DF_FOR, opt, conv)
    d1 = delta(k, VOL, T, F, DF_FOR, opt, conv)
    assert d1 == pytest.approx(d, abs=1e-12)


def test_spot_vs_forward_delta_scaling() -> None:
    k = 1.12
    fd = forward_delta(k, VOL, T, F, OptionType.CALL, premium_adjusted=False)
    sd = delta(k, VOL, T, F, DF_FOR, OptionType.CALL,
               DeltaConvention(DeltaStyle.SPOT, False))
    assert sd == pytest.approx(DF_FOR * fd, abs=1e-15)


class TestPremiumAdjustedBranch:
    """The genuinely dangerous case: PA call delta is non-monotonic in K."""

    def test_delta_is_non_monotonic(self) -> None:
        k_max, d_max = pa_call_delta_max_strike(VOL, T, F)

        def pa(k: float) -> float:
            return forward_delta(k, VOL, T, F, OptionType.CALL, premium_adjusted=True)

        assert pa(k_max * 0.98) < d_max
        assert pa(k_max * 1.02) < d_max
        # delta rises below the turning point and falls above it
        assert pa(k_max * 0.90) < pa(k_max * 0.98)
        assert pa(k_max * 1.02) > pa(k_max * 1.10)

    def test_two_strikes_same_delta_higher_returned(self) -> None:
        """Construct a target hit by TWO strikes; the inversion must return
        the higher-strike (market) branch [RW2010]."""
        conv = DeltaConvention(DeltaStyle.FORWARD, premium_adjusted=True)
        k_max, d_max = pa_call_delta_max_strike(VOL, T, F)
        target = 0.995 * d_max  # just below the peak -> two roots, both near k_max

        k_solved = strike_from_delta(target, VOL, T, F, DF_FOR, OptionType.CALL, conv)
        assert k_solved >= k_max  # correct branch

        # exhibit the OTHER (lower) root explicitly by bisection on the left branch
        from scipy.optimize import brentq

        def pa(k: float) -> float:
            return forward_delta(k, VOL, T, F, OptionType.CALL, premium_adjusted=True)

        k_low = brentq(lambda k: pa(k) - target, 1e-6 * F, k_max)
        assert k_low < k_max < k_solved
        assert pa(k_low) == pytest.approx(pa(k_solved), abs=1e-12)
        assert k_solved != pytest.approx(k_low, rel=1e-3)

    def test_unattainable_delta_raises(self) -> None:
        conv = DeltaConvention(DeltaStyle.FORWARD, premium_adjusted=True)
        _, d_max = pa_call_delta_max_strike(VOL, T, F)
        with pytest.raises(ValueError, match="exceeds the maximum attainable"):
            strike_from_delta(d_max * 1.05, VOL, T, F, DF_FOR, OptionType.CALL, conv)

    def test_pa_put_monotonic_inversion(self) -> None:
        conv = DeltaConvention(DeltaStyle.SPOT, premium_adjusted=True)
        for d in (-0.05, -0.10, -0.25, -0.45):
            k = strike_from_delta(d, VOL, T, F, DF_FOR, OptionType.PUT, conv)
            assert delta(k, VOL, T, F, DF_FOR, OptionType.PUT, conv) == pytest.approx(
                d, abs=1e-12
            )


class TestAtmStrike:
    @pytest.mark.parametrize("pa", [False, True])
    @pytest.mark.parametrize("style", [DeltaStyle.SPOT, DeltaStyle.FORWARD])
    def test_dns_straddle_is_delta_neutral(self, pa: bool, style: DeltaStyle) -> None:
        """The DNS strike must yield a straddle with zero net delta in the
        configured convention -- for all four conventions."""
        conv = DeltaConvention(style, pa)
        k_atm = atm_strike(VOL, T, F, spot=F * 0.99, atm_convention=AtmConvention.DELTA_NEUTRAL,
                           delta_convention=conv)
        dc = delta(k_atm, VOL, T, F, DF_FOR, OptionType.CALL, conv)
        dp = delta(k_atm, VOL, T, F, DF_FOR, OptionType.PUT, conv)
        assert dc + dp == pytest.approx(0.0, abs=1e-14)

    def test_dns_sign_flip(self) -> None:
        """Unadjusted DNS sits ABOVE the forward, premium-adjusted BELOW."""
        conv_u = DeltaConvention(DeltaStyle.SPOT, False)
        conv_pa = DeltaConvention(DeltaStyle.SPOT, True)
        k_u = atm_strike(VOL, T, F, F, AtmConvention.DELTA_NEUTRAL, conv_u)
        k_pa = atm_strike(VOL, T, F, F, AtmConvention.DELTA_NEUTRAL, conv_pa)
        assert k_u == pytest.approx(F * math.exp(0.5 * VOL**2 * T), rel=1e-14)
        assert k_pa == pytest.approx(F * math.exp(-0.5 * VOL**2 * T), rel=1e-14)
        assert k_pa < F < k_u

    def test_atmf_and_spot(self) -> None:
        conv = DeltaConvention(DeltaStyle.SPOT, False)
        assert atm_strike(VOL, T, F, 1.0, AtmConvention.FORWARD, conv) == F
        assert atm_strike(VOL, T, F, 1.0, AtmConvention.SPOT, conv) == 1.0


def test_unattainable_unadjusted_delta_raises() -> None:
    conv = DeltaConvention(DeltaStyle.FORWARD, False)
    with pytest.raises(ValueError):
        strike_from_delta(1.2, VOL, T, F, DF_FOR, OptionType.CALL, conv)
    with pytest.raises(ValueError):
        strike_from_delta(-0.2, VOL, T, F, DF_FOR, OptionType.CALL, conv)
