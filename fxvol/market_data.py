"""Market data model and loaders.

Quote schema per (pair, valuation_date, tenor):

* ``spot`` plus either both zero rates (``r_dom``, ``r_for``) or an outright
  ``forward`` (in which case the missing rate is implied from covered
  interest parity ``F = S exp((r_dom - r_for) T)``; ``r_dom`` is still
  required for discounting).
* ``atm_vol``, ``rr_25``, ``bf_25`` (required); ``rr_10``, ``bf_10``
  (optional).
* Convention metadata falls back to the pair registry
  (:data:`fxvol.conventions.PAIR_REGISTRY`).

File formats: in CSV/JSON files, **vols and rates are in percent** (7.6 means
7.6%, i.e. 0.076); everything is converted to decimals at the IO boundary.
Parsing is kept strictly separate from the math; the engine consumes only
:class:`FxMarketData` / :class:`MarketPillar`.  Any feed (Bloomberg,
Refinitiv, ...) can be plugged in by implementing :class:`QuoteSource`.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

from .conventions import PairConventions, get_pair_conventions

_TENOR_RE = re.compile(r"^(?:(ON|TN|SN)|(\d+)([DWMY]))$", re.IGNORECASE)

#: canonical pillar ordering helper (value = approximate days, only used for
#: sorting labels; actual year fractions come from real dates)
STANDARD_TENORS = ("ON", "1W", "2W", "3W", "1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y")


def add_tenor(start: date, tenor: str) -> date:
    """Expiry date for ``tenor`` from ``start``.

    ON/TN/SN -> 1 calendar day; ``nD``/``nW`` -> calendar days; ``nM``/``nY``
    -> calendar month/year roll with end-of-month clamping.  No holiday
    calendar is applied (a business-day calendar is an extension point; for
    surface construction from illustrative data, calendar-day expiries with
    act/365F are adequate and documented).
    """
    m = _TENOR_RE.match(tenor.strip())
    if not m:
        raise ValueError(f"unrecognised tenor {tenor!r}")
    if m.group(1):
        return start + timedelta(days=1)
    n = int(m.group(2))
    unit = m.group(3).upper()
    if unit == "D":
        return start + timedelta(days=n)
    if unit == "W":
        return start + timedelta(weeks=n)
    months = n * 12 if unit == "Y" else n
    y, mo = divmod((start.year * 12 + start.month - 1) + months, 12)
    mo += 1
    # clamp day to end of target month
    for day in (start.day, 30, 29, 28):
        try:
            return date(y, mo, day)
        except ValueError:
            continue
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class TenorQuote:
    """Raw market quote for one tenor.  Vols/rates as decimals."""

    tenor: str
    atm_vol: float
    rr_25: float
    bf_25: float
    rr_10: float | None = None
    bf_10: float | None = None
    r_dom: float | None = None
    r_for: float | None = None
    forward: float | None = None

    def __post_init__(self) -> None:
        if self.atm_vol <= 0.0:
            raise ValueError(f"{self.tenor}: atm_vol must be positive")
        if (self.rr_10 is None) != (self.bf_10 is None):
            raise ValueError(f"{self.tenor}: rr_10 and bf_10 must come together")

    @property
    def has_10d(self) -> bool:
        return self.rr_10 is not None


@dataclass(frozen=True)
class MarketPillar:
    """A fully-resolved pillar: dates -> year fraction, rates/forward closed
    under covered interest parity.  This is what the engine consumes."""

    tenor: str
    expiry_date: date
    expiry: float  # year fraction (pair's day count)
    forward: float
    df_dom: float
    df_for: float
    r_dom: float
    r_for: float
    quote: TenorQuote


@dataclass(frozen=True)
class FxMarketData:
    """All quotes for one pair on one valuation date."""

    pair: str
    valuation_date: date
    spot: float
    quotes: tuple[TenorQuote, ...]
    conventions: PairConventions

    def pillars(self) -> list[MarketPillar]:
        """Resolve quotes into sorted pillars.

        Rate/forward closure: given ``r_dom`` and ``r_for`` the forward is
        ``F = S exp((r_dom - r_for) T)``; given ``r_dom`` and ``forward`` the
        foreign rate is implied ``r_for = r_dom - ln(F/S)/T`` (forwards are
        preferred as the market observable; the rate differential is what
        matters for the surface).
        """
        out: list[MarketPillar] = []
        dc = self.conventions.day_count
        for q in self.quotes:
            exp_date = add_tenor(self.valuation_date, q.tenor)
            t = dc.year_fraction((exp_date - self.valuation_date).days)
            if t <= 0.0:
                raise ValueError(f"{q.tenor}: non-positive year fraction {t}")
            if q.r_dom is None:
                raise ValueError(
                    f"{q.tenor}: r_dom is required (needed for discounting)"
                )
            r_dom = q.r_dom
            if q.forward is not None:
                fwd = q.forward
                r_for = r_dom - math.log(fwd / self.spot) / t
                if q.r_for is not None and abs(q.r_for - r_for) > 1e-6:
                    raise ValueError(
                        f"{q.tenor}: supplied r_for={q.r_for} inconsistent with "
                        f"forward-implied {r_for:.8f}; supply one or the other"
                    )
            elif q.r_for is not None:
                r_for = q.r_for
                fwd = self.spot * math.exp((r_dom - r_for) * t)
            else:
                raise ValueError(f"{q.tenor}: need r_for or forward")
            out.append(
                MarketPillar(
                    tenor=q.tenor,
                    expiry_date=exp_date,
                    expiry=t,
                    forward=fwd,
                    df_dom=math.exp(-r_dom * t),
                    df_for=math.exp(-r_for * t),
                    r_dom=r_dom,
                    r_for=r_for,
                    quote=q,
                )
            )
        out.sort(key=lambda p: p.expiry)
        for a, b in zip(out, out[1:], strict=False):
            if b.expiry - a.expiry < 1e-10:
                raise ValueError(f"duplicate tenor expiries: {a.tenor}, {b.tenor}")
        return out


class QuoteSource(Protocol):
    """Pluggable data-source seam: any provider adapter (files, Bloomberg,
    Refinitiv, ...) just needs to return an :class:`FxMarketData`."""

    def load(self) -> FxMarketData:  # pragma: no cover - protocol
        ...


def _pct(value: str | float | None) -> float | None:
    """Percent -> decimal at the IO boundary; blank -> None."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        value = float(value)
    return float(value) / 100.0


def _pct_required(value: str | float | None, field: str, tenor: str) -> float:
    x = _pct(value)
    if x is None:
        raise ValueError(f"{tenor}: required quote field {field!r} is missing/blank")
    return x


def _build(
    pair: str,
    valuation_date: date,
    spot: float,
    rows: Iterable[dict],
    conventions: PairConventions | None,
) -> FxMarketData:
    conv = conventions if conventions is not None else get_pair_conventions(pair)
    quotes = tuple(
        TenorQuote(
            tenor=str(r["tenor"]).strip(),
            atm_vol=_pct_required(r["atm_vol"], "atm_vol", str(r["tenor"])),
            rr_25=_pct_required(r["rr_25"], "rr_25", str(r["tenor"])),
            bf_25=_pct_required(r["bf_25"], "bf_25", str(r["tenor"])),
            rr_10=_pct(r.get("rr_10")),
            bf_10=_pct(r.get("bf_10")),
            r_dom=_pct(r.get("r_dom")),
            r_for=_pct(r.get("r_for")),
            forward=(float(r["forward"]) if r.get("forward") not in (None, "") else None),
        )
        for r in rows
    )
    return FxMarketData(
        pair=pair.upper(),
        valuation_date=valuation_date,
        spot=spot,
        quotes=quotes,
        conventions=conv,
    )


def load_csv(
    path: str | Path, conventions: PairConventions | None = None
) -> FxMarketData:
    """Load quotes from CSV.

    Expected columns: ``pair, valuation_date, spot, tenor, atm_vol, rr_25,
    bf_25 [, rr_10, bf_10, r_dom, r_for, forward]``.  Vols/rates in percent;
    spot/forward as outright rates.  Lines starting with ``#`` are comments.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [
            r
            for r in csv.DictReader(
                line for line in fh if not line.lstrip().startswith("#")
            )
        ]
    if not rows:
        raise ValueError(f"{path}: no quote rows found")
    pair = rows[0]["pair"].strip()
    vdate = date.fromisoformat(rows[0]["valuation_date"].strip())
    spot = float(rows[0]["spot"])
    return _build(pair, vdate, spot, rows, conventions)


def load_json(
    path: str | Path, conventions: PairConventions | None = None
) -> FxMarketData:
    """Load quotes from JSON:

    ``{"pair": ..., "valuation_date": "YYYY-MM-DD", "spot": ...,
    "quotes": [{"tenor": ..., "atm_vol": ..., ...}, ...]}``
    with the same percent units as the CSV loader.
    """
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    return _build(
        doc["pair"],
        date.fromisoformat(doc["valuation_date"]),
        float(doc["spot"]),
        doc["quotes"],
        conventions,
    )


@dataclass(frozen=True)
class FileQuoteSource:
    """QuoteSource adapter for local CSV/JSON files."""

    path: Path
    conventions: PairConventions | None = None

    def load(self) -> FxMarketData:
        p = Path(self.path)
        if p.suffix.lower() == ".json":
            return load_json(p, self.conventions)
        return load_csv(p, self.conventions)


def sort_tenors(labels: Sequence[str], valuation_date: date) -> list[str]:
    """Sort tenor labels chronologically using real dates."""
    return sorted(labels, key=lambda t: add_tenor(valuation_date, t))
