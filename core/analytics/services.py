from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any

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

    def on_fill(
        self,
        fill: Fill,
        current_equity: float | None = None,
        current_cash: float | None = None,
    ) -> None:
        """Record fill event and update equity with authoritative wallet snapshot when available."""
        fee = abs(fill.fee)
        if fill.side.upper() == "BUY":
            pnl = -(fill.qty * fill.price) - fee
        else:
            pnl = (fill.qty * fill.price) - fee

        if current_equity is not None:
            prev_equity = self.equity_curve[-1] if self.equity_curve else self.initial_equity
            realized_delta = float(current_equity) - prev_equity
            self._append_bounded(self.trade_pnls, realized_delta)
            self.on_equity(float(current_equity), current_cash=current_cash)
            return

        # Fallback path when no authoritative equity snapshot is provided.
        self._append_bounded(self.trade_pnls, pnl)
        last = self.equity_curve[-1] if self.equity_curve else self.initial_equity
        self.on_equity(last + pnl, current_cash=current_cash)

    def on_equity(self, current_equity: float, current_cash: float | None = None) -> None:
        """Track mark-to-market equity on every market tick for accurate drawdown/sharpe."""
        self._append_bounded(self.equity_curve, float(current_equity))
        if current_cash is not None:
            self._append_bounded(self.cash_curve, float(current_cash))
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
                annualization = math.sqrt(max(self.periods_per_year, 1))
                self.sharpe = (mean_return / std_return) * annualization if std_return > 0 else 0.0
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
    recent_mu: list[float] = field(default_factory=list)
    recent_realized_return: list[float] = field(default_factory=list)
    recent_pred_error: list[float] = field(default_factory=list)
    max_points: int = 500
    ar_direction_hits: int = 0
    ar_samples: int = 0
    ar_sum_abs_error: float = 0.0
    ar_sum_sq_error: float = 0.0
    ar_sum_brier: float = 0.0
    direction_epsilon: float = 5e-5
    _symbol_steps: dict[str, int] = field(default_factory=dict)
    _last_prices: dict[str, float] = field(default_factory=dict)
    _pending_predictions: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def _append_bounded(self, target: list[float], value: float) -> None:
        target.append(value)
        if len(target) > self.max_points:
            del target[:-self.max_points]

    def analyze_prediction(self, pred: Prediction) -> None:
        bucket = "up" if pred.prob_up >= 0.5 else "down"
        self.prediction_distribution[bucket] = self.prediction_distribution.get(bucket, 0) + 1
        self._append_bounded(self.recent_prob_up, float(pred.prob_up))
        self._append_bounded(self.recent_sigma, float(pred.sigma))
        self._append_bounded(self.recent_mu, float(pred.mu))

        if self.recent_prob_up:
            avg_prob = sum(self.recent_prob_up) / len(self.recent_prob_up)
            prob_dispersion = sum(abs(value - 0.5) for value in self.recent_prob_up) / len(self.recent_prob_up)
            self.uncertainty_stats["avg_prob_up"] = avg_prob
            self.uncertainty_stats["avg_distance_from_0_5"] = prob_dispersion
            self.uncertainty_stats["sample_size"] = float(len(self.recent_prob_up))

        if self.recent_sigma:
            self.uncertainty_stats["avg_sigma"] = sum(self.recent_sigma) / len(self.recent_sigma)

    def on_market_price(self, symbol: str, price: float) -> None:
        symbol_key = str(symbol or "UNKNOWN")
        step = self._symbol_steps.get(symbol_key, 0) + 1
        self._symbol_steps[symbol_key] = step

        pending = self._pending_predictions.get(symbol_key, [])
        if pending:
            still_pending: list[dict[str, Any]] = []
            for item in pending:
                target_step = int(item.get("target_step", step + 1))
                start_price = float(item.get("start_price", 0.0))
                if step < target_step:
                    still_pending.append(item)
                    continue

                if start_price <= 0:
                    continue

                realized_ret = (float(price) - start_price) / start_price
                if abs(realized_ret) <= self.direction_epsilon:
                    continue
                pred_mu = float(item.get("mu", 0.0))
                pred_prob_up = min(1.0, max(0.0, float(item.get("prob_up", 0.5))))
                pred_label = 1 if pred_prob_up >= 0.5 else 0
                true_label = 1 if realized_ret > 0 else 0

                self._append_bounded(self.recent_realized_return, realized_ret)
                self._append_bounded(self.recent_pred_error, pred_mu - realized_ret)
                self.ar_samples += 1
                if pred_label == true_label:
                    self.ar_direction_hits += 1
                self.ar_sum_abs_error += abs(pred_mu - realized_ret)
                self.ar_sum_sq_error += (pred_mu - realized_ret) ** 2
                self.ar_sum_brier += (pred_prob_up - float(true_label)) ** 2

            if still_pending:
                self._pending_predictions[symbol_key] = still_pending
            else:
                self._pending_predictions.pop(symbol_key, None)

        self._last_prices[symbol_key] = float(price)

    def track_prediction(self, symbol: str, pred: Prediction) -> None:
        self.analyze_prediction(pred)
        symbol_key = str(symbol or "UNKNOWN")
        horizon = max(int(pred.horizon), 1)
        # Cap AR evaluation at 24 bars for responsive live feedback.
        # Full model horizon (e.g. 96 bars = 8h at 5m) would delay first results by hours.
        ar_eval_horizon = min(horizon, 24)
        current_step = self._symbol_steps.get(symbol_key, 0)
        start_price = self._last_prices.get(symbol_key)
        if start_price is None:
            return

        item = {
            "target_step": current_step + ar_eval_horizon,
            "start_price": float(start_price),
            "prob_up": float(pred.prob_up),
            "mu": float(pred.mu),
        }
        queue = self._pending_predictions.setdefault(symbol_key, [])
        queue.append(item)
        if len(queue) > self.max_points:
            del queue[:-self.max_points]

    def compute_drift(self) -> float:
        return self.feature_drift_score

    def ar_metrics(self) -> dict[str, float]:
        if self.ar_samples <= 0:
            return {
                "samples": 0.0,
                "directional_accuracy": 0.0,
                "mae_return": 0.0,
                "rmse_return": 0.0,
                "brier_up": 0.0,
            }
        samples = float(self.ar_samples)
        return {
            "samples": samples,
            "directional_accuracy": float(self.ar_direction_hits) / samples,
            "mae_return": self.ar_sum_abs_error / samples,
            "rmse_return": math.sqrt(max(self.ar_sum_sq_error / samples, 0.0)),
            "brier_up": self.ar_sum_brier / samples,
        }

    def summary(self) -> dict:
        return {
            "feature_drift_score": self.feature_drift_score,
            "prediction_distribution": self.prediction_distribution,
            "uncertainty_stats": self.uncertainty_stats,
            "recent_prob_up": self.recent_prob_up,
            "recent_sigma": self.recent_sigma,
            "recent_mu": self.recent_mu,
            "recent_realized_return": self.recent_realized_return,
            "recent_pred_error": self.recent_pred_error,
            "ar_metrics": self.ar_metrics(),
        }
