"""The stock universe: which firms the study uses, and where their data are found.

``configs/universe.yaml`` lists each stock under the ticker it had in January 2015,
which identifies it everywhere in the study (file names, seeds). Each entry gives:

* ``ciks``: every SEC CIK that filed the firm's accounts from 2014 to 2025; a
  holding-company reorganisation (Google into Alphabet, for example) adds one;
* ``yahoo``: the Yahoo Finance symbol of its price history today, or null when
  Yahoo no longer has it;
* ``role``: ``main`` (development and test years), ``holdout`` (final campaign
  only), ``candidate`` (not yet assigned) or ``excluded`` (unusable; ``notes``
  give the reason);
* ``acttrade``: whether it is one of ActTrade's 18 US stocks (null until known).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from regime_fusion.config import ConfigError

DEFAULT_UNIVERSE = Path(__file__).resolve().parents[3] / "configs" / "universe.yaml"
ROLES = ("main", "holdout", "candidate", "excluded")
_ID = re.compile(r"[A-Z][A-Z.\-]{0,6}")


@dataclass(frozen=True)
class Stock:
    id: str
    name: str
    sector: str
    ciks: tuple[int, ...]
    yahoo: str | None
    role: str
    acttrade: bool | None = None
    notes: str = ""


@dataclass(frozen=True)
class Universe:
    stocks: tuple[Stock, ...]

    def __post_init__(self) -> None:
        for stock in self.stocks:
            _validate(stock)
        _unique([s.id for s in self.stocks], "stock id")
        _unique([s.yahoo for s in self.stocks if s.yahoo], "Yahoo symbol")
        flagged = sum(bool(s.acttrade) for s in self.stocks)
        if flagged > 18:
            raise ConfigError(f"{flagged} stocks flagged as ActTrade's; it has 18 US stocks")

    def get(self, stock_id: str) -> Stock:
        for stock in self.stocks:
            if stock.id == stock_id:
                return stock
        raise KeyError(stock_id)

    def select(self, *roles: str, include_holdout: bool = False) -> list[Stock]:
        """Stocks with any of ``roles``, in file order.

        Hold-out stocks stay untouched until the final campaign, so they are
        returned only with ``include_holdout=True``: pass it only to download
        data or to run the final campaign.
        """
        unknown = set(roles) - set(ROLES)
        if unknown:
            raise ValueError(f"unknown roles {sorted(unknown)}")
        if "holdout" in roles and not include_holdout:
            raise ValueError(
                "hold-out stocks are untouched during development; "
                "pass include_holdout=True only for downloads or the final campaign"
            )
        return [stock for stock in self.stocks if stock.role in roles]


def load_universe(path: str | Path = DEFAULT_UNIVERSE) -> Universe:
    """Read and validate the universe file."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("stocks"), list):
        raise ConfigError(f"{path}: expected a 'stocks' list")
    stocks = []
    for entry in raw["stocks"]:
        try:
            stocks.append(
                Stock(
                    id=str(entry["id"]),
                    name=str(entry["name"]),
                    sector=str(entry["sector"]),
                    ciks=tuple(int(cik) for cik in entry["ciks"]),
                    yahoo=None if entry.get("yahoo") is None else str(entry["yahoo"]),
                    role=str(entry["role"]),
                    acttrade=entry.get("acttrade"),
                    notes=str(entry.get("notes") or ""),
                )
            )
        except (KeyError, TypeError) as problem:
            raise ConfigError(f"{path}: bad stock entry {entry!r} ({problem})") from None
    return Universe(tuple(stocks))


def _validate(stock: Stock) -> None:
    if not _ID.fullmatch(stock.id):
        raise ConfigError(f"{stock.id!r}: a stock id is its January 2015 ticker, e.g. 'BRK.B'")
    if stock.role not in ROLES:
        raise ConfigError(f"{stock.id}: role must be one of {ROLES}")
    if not stock.ciks or any(cik <= 0 for cik in stock.ciks):
        raise ConfigError(f"{stock.id}: needs at least one positive CIK")
    if stock.role in ("main", "holdout") and not stock.yahoo:
        raise ConfigError(f"{stock.id}: a {stock.role} stock needs a Yahoo symbol")
    if stock.acttrade not in (None, True, False):
        raise ConfigError(f"{stock.id}: acttrade must be true, false or null")
    if stock.role == "excluded" and not stock.notes:
        raise ConfigError(f"{stock.id}: say in notes why the stock is excluded")


def _unique(values: list[str], label: str) -> None:
    seen, repeated = set(), set()
    for value in values:
        (repeated if value in seen else seen).add(value)
    if repeated:
        raise ConfigError(f"repeated {label}: {sorted(repeated)}")
