from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math

from core.domain.models import Fill
from core.ml.services import Prediction


# Dashboardhoz exportálható snapshot.
@dataclass(slots=True)
class PerformanceSnapshot:
    equity: float
    drawdown: float
    sharpe: float
    win_rate: float
    positions: dict
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Egyszerű performance tracker.
@dataclass(slots=True)
class PerformanceTracker:
    # Configuration
    initial_equity: float = 10000.0
    max_points: int = 5000
    risk_free_rate_annual: float = 0.0
    periods_per_year: int = 252

    # Time-series state
    equity_curve: list[float] = field(default_factory=list)
    cash_curve: list[float] = field(default_factory=list)
    trade_pnls: list[float] = field(default_factory=list)
    peak_equity: float = 0.0

    # Derived statistics
    drawdown: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.0
    avg_trade: float = 0.0
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.peak_equity <= 0:
            self.peak_equity = self.initial_equity

    def _append_bounded(self, target: list[float], value: float) -> None:
        target.append(value)
        if len(target) > self.max_points:
            del target[:-self.max_points]

    def on_fill(self, fill: Fill, current_equity: float | None = None) -> None:
        """Record fill event with actual equity impact from wallet."""
        fee = abs(fill.fee)
        if fill.side.upper() == "BUY":
            pnl = -(fill.qty * fill.price) - fee
        else:
            pnl = (fill.qty * fill.price) - fee
        self._append_bounded(self.trade_pnls, pnl)

        last = self.equity_curve[-1] if self.equity_curve else self.initial_equity
        next_equity = last + pnl
        self._append_bounded(self.equity_curve, next_equity)

        if current_equity is not None:
            self._append_bounded(self.cash_curve, current_equity)

        self._update_statistics()

    def on_prediction(self, pred: Prediction) -> None:
        # MVP: csak placeholder, itt lehetne diagnosztikát gyűjteni.
        _ = pred

    def _update_statistics(self) -> None:
        if not self.equity_curve:
            self.drawdown = 0.0
            self.sharpe = 0.0
            self.win_rate = 0.0
            self.avg_trade = 0.0
            return

        current_equity = self.equity_curve[-1]
        self.peak_equity = max(self.peak_equity, current_equity)
        if self.peak_equity > 0:
            self.drawdown = ((self.peak_equity - current_equity) / self.peak_equity) * 100.0
        else:
            self.drawdown = 0.0

        trade_count = len(self.trade_pnls)
        if trade_count > 0:
            wins = sum(1 for value in self.trade_pnls if value > 0)
            self.win_rate = (wins / trade_count) * 100.0
            self.avg_trade = sum(self.trade_pnls) / trade_count
        else:
            self.win_rate = 0.0
            self.avg_trade = 0.0

        if len(self.equity_curve) >= 2:
            returns: list[float] = []
            prev = self.initial_equity
            for current in self.equity_curve:
                if prev != 0:
                    returns.append((current - prev) / prev)
                prev = current

            if len(returns) >= 2:
                rf_per_period = self.risk_free_rate_annual / max(self.periods_per_year, 1)
                excess_returns = [value - rf_per_period for value in returns]
                mean_return = sum(excess_returns) / len(excess_returns)
                variance = sum((value - mean_return) ** 2 for value in excess_returns) / (len(excess_returns) - 1)
                std_return = math.sqrt(max(variance, 0.0))
                self.sharpe = (mean_return / std_return) * math.sqrt(len(excess_returns)) if std_return > 0 else 0.0
            else:
                self.sharpe = 0.0
        else:
            self.sharpe = 0.0

    def snapshot(self) -> PerformanceSnapshot:
        equity = self.equity_curve[-1] if self.equity_curve else self.initial_equity
        cash = self.cash_curve[-1] if self.cash_curve else self.initial_equity
        return PerformanceSnapshot(
            equity=equity,
            drawdown=self.drawdown,
            sharpe=self.sharpe,
            win_rate=self.win_rate,
            positions={"equity": equity, "cash": cash},
        )


# Aggregált üzleti metrikák.
@dataclass(slots=True)
class MetricsAggregator:
    counters: dict[str, int] = field(default_factory=dict)
    gauges: dict[str, float] = field(default_factory=dict)

    def increment(self, name: str) -> None:
        self.counters[name] = self.counters.get(name, 0) + 1

    def set_gauge(self, name: str, value: float) -> None:
        self.gauges[name] = value

    def export(self) -> dict:
        return {"counters": self.counters, "gauges": self.gauges}


# Modell diagnosztika szolgáltatás.
@dataclass(slots=True)
class ModelDiagnosticsService:
    feature_drift_score: float = 0.0
    prediction_distribution: dict[str, int] = field(default_factory=dict)
    uncertainty_stats: dict[str, float] = field(default_factory=dict)
    recent_prob_up: list[float] = field(default_factory=list)
    recent_sigma: list[float] = field(default_factory=list)
    max_points: int = 500

    def _append_bounded(self, target: list[float], value: float) -> None:
        target.append(value)
        if len(target) > self.max_points:
            del target[:-self.max_points]

    def analyze_prediction(self, pred: Prediction) -> None:
        bucket = "up" if pred.prob_up >= 0.5 else "down"
        self.prediction_distribution[bucket] = self.prediction_distribution.get(bucket, 0) + 1
        self._append_bounded(self.recent_prob_up, float(pred.prob_up))
        self._append_bounded(self.recent_sigma, float(pred.sigma))

        if self.recent_prob_up:
            avg_prob = sum(self.recent_prob_up) / len(self.recent_prob_up)
            prob_dispersion = sum(abs(value - 0.5) for value in self.recent_prob_up) / len(self.recent_prob_up)
            self.uncertainty_stats["avg_prob_up"] = avg_prob
            self.uncertainty_stats["avg_distance_from_0_5"] = prob_dispersion
            self.uncertainty_stats["sample_size"] = float(len(self.recent_prob_up))

        if self.recent_sigma:
            self.uncertainty_stats["avg_sigma"] = sum(self.recent_sigma) / len(self.recent_sigma)

    def compute_drift(self) -> float:
        return self.feature_drift_score

    def summary(self) -> dict:
        return {
            "feature_drift_score": self.feature_drift_score,
            "prediction_distribution": self.prediction_distribution,
            "uncertainty_stats": self.uncertainty_stats,
            "recent_prob_up": self.recent_prob_up,
        }
