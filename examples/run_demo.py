"""End-to-end demo: quotes -> calibration -> surface -> arbitrage -> plots.

Run from the repo root:

    python examples/run_demo.py [--data examples/eurusd_quotes.csv] [--out examples/output]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from fxvol import OptionType, VolSurface, check_surface, load_csv, load_json
from fxvol.io_report import calibration_report
from fxvol.viz import (
    plot_arbitrage,
    plot_density,
    plot_smiles,
    plot_surface_3d,
    plot_term_structure,
)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(here / "eurusd_quotes.csv"))
    ap.add_argument("--out", default=str(here / "output"))
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    market = load_json(data) if data.suffix == ".json" else load_csv(data)
    print(f"Loaded {market.pair} quotes: {[q.tenor for q in market.quotes]}")

    surface = VolSurface.from_market(market)
    arb = check_surface(surface)

    report = calibration_report(market, surface, arb)
    print(report)
    (out / "calibration_report.txt").write_text(report, encoding="utf-8")

    # a few sample queries
    t = 0.5
    k_atm = surface.forward(t)
    print(f"\nSample queries at T={t}:")
    print(f"  forward           : {k_atm:.6f}")
    print(f"  ATM-forward vol   : {surface.vol(k_atm, t) * 100:.4f}%")
    k25, v25 = surface.vol_from_delta(0.25, t, OptionType.CALL)
    print(f"  25d call          : K={k25:.6f}  vol={v25 * 100:.4f}%")
    g = surface.greeks(k25, t, OptionType.CALL)
    print(f"  25d call greeks   : delta_spot={g.delta_spot:.4f} vega={g.vega:.6f} "
          f"vanna={g.vanna:.6f} volga={g.volga:.6f}")

    plot_surface_3d(surface, save=out / "surface_3d.png")
    plot_smiles(surface, save=out / "smiles.png")
    plot_term_structure(market, save=out / "term_structure.png")
    plot_density(surface, "1Y", save=out / "density_1y.png")
    plot_arbitrage(arb, save=out / "arbitrage_g.png")
    print(f"\nPlots + report written to {out}")


if __name__ == "__main__":
    main()
