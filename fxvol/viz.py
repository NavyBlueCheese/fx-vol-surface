"""Visualisation: 3D surface, smile slices, RR/BF term structure, implied
density, arbitrage diagnostics.  Matplotlib only (plotly optional upstream).

Every function takes ``save`` (path) and returns the Figure; call
``matplotlib.use("Agg")`` upstream for headless use.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .arbitrage import ArbitrageReport
from .conventions import OptionType
from .market_data import FxMarketData
from .pricing import forward_price
from .surface import VolSurface


def _finish(fig: Figure, save: str | Path | None) -> Figure:
    fig.tight_layout()
    if save is not None:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=140)
    return fig


def plot_surface_3d(
    surface: VolSurface,
    n_k: int = 61,
    n_t: int = 40,
    save: str | Path | None = None,
) -> Figure:
    """3D implied vol surface over (delta-ish log-moneyness, T)."""
    t_lo, t_hi = surface.expiries[0], surface.expiries[-1]
    ts = np.linspace(t_lo, t_hi, n_t)
    # k range from the longest pillar's node span
    last = surface.pillars[-1]
    ks_nodes = [np.log(n.strike / last.forward) for n in last.smile.nodes if np.isfinite(n.strike)]
    k_hi = max(ks_nodes) * 1.1 if ks_nodes else 0.3
    k_lo = min(ks_nodes) * 1.1 if ks_nodes else -0.3
    ks = np.linspace(k_lo, k_hi, n_k)
    K, T = np.meshgrid(ks, ts)
    V = np.empty_like(K)
    for i, t in enumerate(ts):
        w = np.asarray(surface.total_variance(ks, float(t)), dtype=float)
        V[i, :] = np.sqrt(w / t) * 100.0
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(K, T, V, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel("forward log-moneyness k = ln(K/F)")
    ax.set_ylabel("expiry T (y)")
    ax.set_zlabel("implied vol (%)")
    ax.set_title(f"{surface.conventions.pair} implied vol surface")
    return _finish(fig, save)


def plot_smiles(
    surface: VolSurface,
    tenors: Sequence[str] | None = None,
    n: int = 121,
    save: str | Path | None = None,
) -> Figure:
    """Smile slices per tenor with calibrated nodes marked."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pillars = surface.pillars if tenors is None else [surface.pillar(t) for t in tenors]
    for p in pillars:
        strikes_nodes = [nd.strike for nd in p.smile.nodes if np.isfinite(nd.strike)]
        lo = min(strikes_nodes) * 0.99
        hi = max(strikes_nodes) * 1.01
        kk = np.linspace(lo, hi, n)
        vv = np.asarray(p.smile.vol(kk), dtype=float) * 100
        (line,) = ax.plot(kk, vv, label=p.tenor)
        ax.plot(
            strikes_nodes,
            [nd.vol * 100 for nd in p.smile.nodes if np.isfinite(nd.strike)],
            "o", ms=5, color=line.get_color(),
        )
    ax.set_xlabel("strike")
    ax.set_ylabel("implied vol (%)")
    ax.set_title(f"{surface.conventions.pair} smiles (markers = calibrated nodes)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _finish(fig, save)


def plot_term_structure(
    market: FxMarketData, save: str | Path | None = None
) -> Figure:
    """ATM / RR / BF quote term structures."""
    pillars = market.pillars()
    ts = [p.expiry for p in pillars]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(ts, [p.quote.atm_vol * 100 for p in pillars], "o-")
    axes[0].set_title("ATM vol (%)")
    axes[1].plot(ts, [p.quote.rr_25 * 100 for p in pillars], "o-", label="25d")
    axes[1].plot(
        ts,
        [p.quote.rr_10 * 100 if p.quote.rr_10 is not None else np.nan for p in pillars],
        "s--", label="10d",
    )
    axes[1].set_title("Risk reversal (%)")
    axes[1].legend()
    axes[2].plot(ts, [p.quote.bf_25 * 100 for p in pillars], "o-", label="25d")
    axes[2].plot(
        ts,
        [p.quote.bf_10 * 100 if p.quote.bf_10 is not None else np.nan for p in pillars],
        "s--", label="10d",
    )
    axes[2].set_title("Butterfly (%)")
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel("T (y)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{market.pair} quote term structures")
    return _finish(fig, save)


def plot_density(
    surface: VolSurface,
    tenor: str,
    n: int = 401,
    save: str | Path | None = None,
) -> Figure:
    """Implied risk-neutral density d2C/dK2 (undiscounted) for one tenor."""
    p = surface.pillar(tenor)
    strikes_nodes = [nd.strike for nd in p.smile.nodes if np.isfinite(nd.strike)]
    lo = min(strikes_nodes) * 0.95
    hi = max(strikes_nodes) * 1.05
    kk = np.linspace(lo, hi, n)
    vols = np.asarray(p.smile.vol(kk), dtype=float)
    calls = np.asarray(
        forward_price(p.forward, kk, vols, p.expiry, 1.0, OptionType.CALL), dtype=float
    )
    dens = np.gradient(np.gradient(calls, kk), kk)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(kk, dens)
    ax.axhline(0.0, color="k", lw=0.8)
    for s in strikes_nodes:
        ax.axvline(s, color="grey", lw=0.6, alpha=0.5)
    ax.set_xlabel("strike")
    ax.set_ylabel("density d2C/dK2")
    ax.set_title(f"{surface.conventions.pair} {p.tenor} implied density (grid FD)")
    ax.grid(alpha=0.3)
    return _finish(fig, save)


def plot_arbitrage(
    report: ArbitrageReport, save: str | Path | None = None
) -> Figure:
    """Durrleman g(k) per tenor; negative regions are butterfly arbitrage."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for tenor, (k, g) in report.durrleman.items():
        ax.plot(k[2:-2], g[2:-2], label=tenor)
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xlabel("forward log-moneyness k")
    ax.set_ylabel("Durrleman g(k)")
    ax.set_title("Butterfly-arbitrage diagnostic: g(k) must stay >= 0")
    ax.legend()
    ax.grid(alpha=0.3)
    return _finish(fig, save)
