"""Static no-arbitrage diagnostics: butterfly (convexity/density) and
calendar checks.

These are *diagnostics, not silent fixes*: violations are reported with
their location; quotes are never altered.

Butterfly / convexity at fixed T
--------------------------------
1. Undiscounted call prices ``C(K)`` must be non-increasing and convex in K;
   ``dC/dK in [-1, 0]`` and ``d2C/dK2 >= 0`` -- the second derivative is the
   (undiscounted) risk-neutral density.
2. The rigorous total-variance condition (Durrleman; Gatheral & Jacquier
   2014, eq. 2.1): with ``w(k)`` total variance at log-moneyness k,

   ``g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2  >=  0``.

   ``g(k) >= 0`` (plus a vanishing-time-value wing condition) is equivalent
   to a non-negative implied density.

Calendar
--------
Total variance ``w(k, T)`` must be non-decreasing in T at constant forward
log-moneyness k [Gatheral 2006].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .conventions import OptionType
from .surface import VolSurface


@dataclass(frozen=True)
class Violation:
    """One arbitrage violation, located in (tenor/T, k/strike) space."""

    kind: str  # "butterfly.monotonicity" | "butterfly.convexity" |
    #            "butterfly.durrleman" | "calendar"
    tenor: str
    expiry: float
    log_moneyness: float
    strike: float
    value: float  # magnitude of the violation (units depend on kind)
    detail: str = ""


@dataclass
class ArbitrageReport:
    """Aggregated diagnostics for a surface."""

    butterfly: list[Violation] = field(default_factory=list)
    calendar: list[Violation] = field(default_factory=list)
    #: grids retained for plotting: {tenor: (k, g(k))}
    durrleman: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.butterfly and not self.calendar

    def summary(self) -> str:
        lines = []
        status = "PASS" if self.ok else "FAIL"
        lines.append(f"Arbitrage check: {status}")
        lines.append(
            f"  butterfly violations: {len(self.butterfly)}; "
            f"calendar violations: {len(self.calendar)}"
        )
        for v in (self.butterfly + self.calendar)[:20]:
            lines.append(
                f"  [{v.kind}] {v.tenor} T={v.expiry:.4f} k={v.log_moneyness:+.4f} "
                f"K={v.strike:.6f} value={v.value:.3e} {v.detail}"
            )
        if len(self.butterfly) + len(self.calendar) > 20:
            lines.append(f"  ... and {len(self.butterfly) + len(self.calendar) - 20} more")
        return "\n".join(lines)


def _pillar_k_grid(surface: VolSurface, i: int, n: int, pad: float) -> np.ndarray:
    """Grid in k spanning the pillar's node range padded multiplicatively.

    The grid deliberately stays inside/near the quoted region plus a margin;
    the flat-vol extrapolation wings have a known benign kink at the last
    node which we cover with ``pad``.
    """
    p = surface.pillars[i]
    smile = p.smile
    nodes = smile.nodes
    strikes = [n_.strike for n_ in nodes if np.isfinite(n_.strike)]
    if strikes:
        k_lo = np.log(min(strikes) / p.forward)
        k_hi = np.log(max(strikes) / p.forward)
    else:  # smile without strike nodes (e.g. Malz): use +/-3 sigma sqrt(T)
        s = float(np.asarray(smile.vol(p.forward), dtype=float))
        k_hi = 3.0 * s * np.sqrt(p.expiry)
        k_lo = -k_hi
    span = k_hi - k_lo
    return np.linspace(k_lo - pad * span, k_hi + pad * span, n)


def check_butterfly(
    surface: VolSurface,
    n_grid: int = 201,
    pad: float = 0.25,
    price_tol: float = 1e-10,
    g_tol: float = 1e-8,
) -> tuple[list[Violation], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Convexity/monotonicity of undiscounted C(K) + Durrleman g(k) >= 0 per
    pillar.  Derivatives via central finite differences on a uniform k grid.

    Tolerances absorb finite-difference noise; they are far below any
    economically meaningful violation.
    """
    violations: list[Violation] = []
    g_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, p in enumerate(surface.pillars):
        k = _pillar_k_grid(surface, i, n_grid, pad)
        strikes = p.forward * np.exp(k)
        vols = np.asarray(p.smile.vol(strikes), dtype=float)
        # undiscounted call prices (forward measure)
        from .pricing import forward_price  # local import avoids cycle at module load

        calls = np.asarray(
            forward_price(p.forward, strikes, vols, p.expiry, 1.0, OptionType.CALL),
            dtype=float,
        )
        dk = np.diff(strikes)
        slope = np.diff(calls) / dk
        # monotonicity: slope in [-1, 0] (undiscounted)
        for j, s in enumerate(slope):
            if s > price_tol or s < -1.0 - 1e-9:
                violations.append(
                    Violation(
                        "butterfly.monotonicity", p.tenor, p.expiry,
                        float(0.5 * (k[j] + k[j + 1])),
                        float(0.5 * (strikes[j] + strikes[j + 1])),
                        float(s),
                        "dC/dK outside [-1, 0]",
                    )
                )
        # convexity: second difference >= 0  (≈ density)
        curv = 2.0 * np.diff(slope) / (strikes[2:] - strikes[:-2])
        for j, c in enumerate(curv):
            if c < -price_tol:
                violations.append(
                    Violation(
                        "butterfly.convexity", p.tenor, p.expiry,
                        float(k[j + 1]), float(strikes[j + 1]), float(c),
                        "d2C/dK2 < 0 (negative density)",
                    )
                )
        # Durrleman g(k) via FD on w(k)
        w = vols * vols * p.expiry
        dk_u = k[1] - k[0]
        wp = np.gradient(w, dk_u, edge_order=2)
        wpp = np.gradient(wp, dk_u, edge_order=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = (
                (1.0 - k * wp / (2.0 * w)) ** 2
                - 0.25 * wp * wp * (1.0 / w + 0.25)
                + 0.5 * wpp
            )
        g_curves[p.tenor] = (k, g)
        interior = slice(2, -2)  # FD edges are noisy
        for kk, gg, ss in zip(k[interior], g[interior], strikes[interior], strict=True):
            if gg < -g_tol:
                violations.append(
                    Violation(
                        "butterfly.durrleman", p.tenor, p.expiry,
                        float(kk), float(ss), float(gg), "g(k) < 0",
                    )
                )
    return violations, g_curves


def check_calendar(
    surface: VolSurface,
    n_grid: int = 101,
    tol: float = 1e-10,
) -> list[Violation]:
    """w(k, T) non-decreasing in T at constant k across adjacent pillars."""
    violations: list[Violation] = []
    if len(surface.pillars) < 2:
        return violations
    # common k grid: union of node spans
    k_los, k_his = [], []
    for i in range(len(surface.pillars)):
        grid = _pillar_k_grid(surface, i, 3, 0.0)
        k_los.append(grid[0])
        k_his.append(grid[-1])
    k = np.linspace(min(k_los), max(k_his), n_grid)
    w_prev = None
    for p in surface.pillars:
        strikes = p.forward * np.exp(k)
        vols = np.asarray(p.smile.vol(strikes), dtype=float)
        w = vols * vols * p.expiry
        if w_prev is not None:
            bad = w - w_prev < -tol
            for kk, wv, wp_, ss in zip(
                k[bad], w[bad], np.asarray(w_prev)[bad], strikes[bad], strict=True
            ):
                violations.append(
                    Violation(
                        "calendar", p.tenor, p.expiry, float(kk), float(ss),
                        float(wv - wp_),
                        f"w decreases from previous pillar ({wp_:.6e} -> {wv:.6e})",
                    )
                )
        w_prev = w
    return violations


def check_surface(
    surface: VolSurface,
    n_grid_butterfly: int = 201,
    n_grid_calendar: int = 101,
) -> ArbitrageReport:
    """Run all static no-arbitrage diagnostics and aggregate a report."""
    bfly, g_curves = check_butterfly(surface, n_grid=n_grid_butterfly)
    cal = check_calendar(surface, n_grid=n_grid_calendar)
    return ArbitrageReport(butterfly=bfly, calendar=cal, durrleman=g_curves)
