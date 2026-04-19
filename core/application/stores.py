from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.application.ports import IStateRepository, ITimeSource
from core.domain.models import Fill, Instrument, Position
from core.ml.services import EstimatedState


# Egyszeru konfiguracio value object.
@dataclass(slots=True)
class Config:
    env: str = "dev"
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])
    risk_limits: dict[str, Any] = field(
        default_factory=lambda: {
            "max_qty": 1.0,
            "min_confidence": 0.5,
            "max_notional": 1000.0,
            "max_abs_position_qty": 2.0,
            "max_slippage_bps": 50.0,
            "allow_short_selling": True,
            "estimated_fee_bps": 10.0,
        }
    )
    broker: dict[str, Any] = field(default_factory=dict)
    bus: dict[str, Any] = field(default_factory=dict)
    storage: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] = field(
        default_factory=lambda: {
            "threshold": 0.55,
            "min_edge": 0.02,
            "min_confidence": 0.55,
            "min_expected_value": 0.0,
            "min_signal_score": 0.0,
            "max_sigma": 1.0,
            "min_sigma": 1e-8,
            "max_strength": 0.0,
            "use_edge_score": False,
            "min_cooldown_seconds": 0.0,
            "min_bars_between_signals": 0,
            "allowed_horizons": [],
            "horizon_scale_mode": "inverse",
            "flip_extra_entry": 0.0,
        }
    )
    simulation: dict[str, Any] = field(default_factory=lambda: {"initial_cash": 10000.0})
    web: dict[str, Any] = field(default_factory=lambda: {"host": "0.0.0.0", "port": int(os.getenv("APP_PORT", "8080")), "hold_open_seconds": 120})

    def get(self, path: str, default: Any = None) -> Any:
        # Egyszeru dotted-path getter.
        tokens = path.split(".")
        if not tokens:
            return default

        current: Any
        first = tokens[0]
        if hasattr(self, first):
            current = getattr(self, first)
        else:
            return default

        for token in tokens[1:]:
            if isinstance(current, dict) and token in current:
                current = current[token]
            else:
                return default
        return current

    def validate(self) -> None:
        if not self.symbols:
            raise ValueError("Legalabb egy symbol szukseges")


# State store: estimator outputok tarolasa.
@dataclass(slots=True)
class StateStore(IStateRepository):
    states: dict[str, EstimatedState] = field(default_factory=dict)

    def save(self, instrument: Instrument, state: EstimatedState) -> None:
        self.states[instrument.symbol] = state

    def update(self, instrument: Instrument, state: EstimatedState) -> None:
        self.save(instrument, state)

    def get(self, instrument: Instrument) -> EstimatedState:
        state = self.states.get(instrument.symbol)
        if state is None:
            raise KeyError(f"No state for instrument '{instrument.symbol}'. Call save() first.")
        return state


# Position store: fill-ekbol pozicio frissitese.
@dataclass(slots=True)
class PositionStore:
    positions: dict[str, Position] = field(default_factory=dict)

    def apply_fill(self, fill: Fill, instrument: Instrument) -> None:
        pos = self.positions.setdefault(instrument.symbol, Position())
        pos.apply_fill(fill)

    def get(self, instrument: Instrument) -> Position:
        return self.positions.setdefault(instrument.symbol, Position())

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            symbol: {
                "qty": value.qty,
                "avg_price": value.avg_price,
                "realized_pnl": value.realized_pnl,
                "unrealized_pnl": value.unrealized_pnl,
            }
            for symbol, value in self.positions.items()
        }


# Lokalis paper-trading wallet: minden futasnal resetelheto kezdotokevel.
@dataclass(slots=True)
class SimulationWallet:
    initial_cash: float = 10000.0
    cash: float | None = field(default=None)
    last_prices: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, float | str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cash is None:
            self.cash = self.initial_cash

    def reset(self) -> None:
        self.cash = self.initial_cash
        self.last_prices = {}
        self.history = []

    def mark_price(self, instrument: Instrument, price: float) -> None:
        price_value = float(price)
        if price_value <= 0:
            raise ValueError("mark price must be > 0")
        self.last_prices[instrument.symbol] = price_value

    def apply_fill(self, fill: Fill, instrument: Instrument) -> None:
        price = float(fill.price)
        qty = float(fill.qty)
        fee = float(fill.fee)
        if fill.side.upper() == "BUY":
            self.cash -= qty * price + fee
        else:
            self.cash += qty * price - fee
        self.last_prices.setdefault(instrument.symbol, price)

    def equity(self, position_store: PositionStore) -> float:
        market_value = 0.0
        for symbol, position in position_store.positions.items():
            price = self.last_prices.get(symbol, position.avg_price)
            market_value += position.qty * price
        return self.cash + market_value

    def snapshot(self, position_store: PositionStore) -> dict[str, float]:
        equity_value = self.equity(position_store)
        return {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "equity": equity_value,
            "pnl": equity_value - self.initial_cash,
        }

    def record(self, position_store: PositionStore, ts: datetime | None = None) -> None:
        # Mark all positions to market before snapshotting so unrealized_pnl stays current.
        for symbol, position in position_store.positions.items():
            price = self.last_prices.get(symbol, position.avg_price)
            position.mark_to_market(price)
        snap = self.snapshot(position_store)
        event_ts = ts if ts is not None else datetime.now(timezone.utc)
        self.history.append(
            {
                "ts": event_ts.astimezone(timezone.utc).isoformat(),
                "cash": float(snap["cash"]),
                "equity": float(snap["equity"]),
                "pnl": float(snap["pnl"]),
            }
        )
        if len(self.history) > 2000:
            self.history = self.history[-2000:]


# Futasi kontextus, amit a stage-ek visznek magukkal.
@dataclass(slots=True)
class RunContext:
    correlation_id: str
    instrument: Instrument
    state_store: IStateRepository
    position_store: PositionStore
    wallet: SimulationWallet
    config: Config
    time_source: ITimeSource
    bar_index: int = 0

    def now(self) -> datetime:
        return self.time_source.now()
