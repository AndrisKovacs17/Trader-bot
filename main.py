from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import os

# Configure logging to see pipeline debug messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
# Suppress noisy HTTP debug logs from the binance client
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

from adapters.infrastructure.binance_feed import (
    BinanceWebSocketStream,
    get_historical_klines_time_window,
    get_historical_klines_with_taker,
    get_historical_funding_rates,
    parse_kline_message,
    to_price_points,
)
from adapters.infrastructure.event_bus import AsyncInMemoryEventBus, DataRecorder, EventStoreImpl
from adapters.infrastructure.event_mapping import iter_news_df_to_events, iter_price_points_to_market_events
from adapters.infrastructure.execution import (
    ExecutionOrderEventHandler,
)
from adapters.testing.broker import MockBrokerGateway
from adapters.infrastructure.news_feed import add_sentiment_columns, get_all_news, NewsStateService
from adapters.infrastructure.time_source import SystemTimeSource
from adapters.observability.event_tap import EventTapHandler
from adapters.offline_training.training_engine import TrainingEngine
from adapters.web.dashboard import DashboardAPI, HttpServer, WebSocketStreamer
from core.analytics.services import MetricsAggregator, ModelDiagnosticsService, PerformanceTracker
from core.application.engine import TradingEngine
from core.application.execution import ExecutionUseCase
from core.application.stages import ExecutionStage, PredictionStage, RiskStage, SignalStage, StateEstimationStage
from core.application.stores import Config, PositionStore, SimulationWallet, StateStore
from core.domain.events import Event
from core.domain.models import Instrument
from core.domain.risk import ConfidenceRule, MaxAbsPositionRule, MaxNotionalRule, MaxQtyRule, NewsSentimentGateRule, RiskPolicy, SlippageGuardRule
from core.domain.strategy import ThresholdStrategy
from core.domain.events import MarketDataEvent
from core.ml.services import KLAPredictor, KLAStateEstimator
from core.ops.contracts import HealthStatus
from core.ops.contracts import SimpleMetrics


# =====================================================
# HEXAGONAL WIRING: Core + Infrastructure + Adapters
# =====================================================


def dashboard_health(config: Config) -> HealthStatus:
    return HealthStatus(ok=True, details={"env": config.env})


def simulation_snapshot(
    wallet: SimulationWallet,
    position_store: PositionStore,
    perf_tracker: PerformanceTracker,
    policy: RiskPolicy,
) -> dict:
    performance = perf_tracker.snapshot()
    return {
        "wallet": wallet.snapshot(position_store),
        "positions": position_store.snapshot(),
        "performance": {
            "equity": performance.equity,
            "drawdown": performance.drawdown,
            "sharpe": performance.sharpe,
            "win_rate": performance.win_rate,
            "positions": performance.positions,
            "timestamp": performance.timestamp,
        },
        "risk": policy.diagnostics(),
    }


async def run_demo() -> None:
    """
    Main demo: Full hexagonal architecture wiring with:
    - Core domain (strategy, risk policy)
    - Application layer (pipeline stages, TradingEngine)
    - Infrastructure adapters (event bus, broker gateway)
    - Offline training (TrainingEngine)
    - Observability (metrics, dashboard)
    """
    
    # =====================================================
    # 1. CONFIGURATION & SHARED INFRASTRUCTURE
    # =====================================================
    
    config = Config(
        env="dev",
        symbols=["BTCUSDT"],
        model={
            "signal_threshold": 0.51,
            "max_abs_mu": 0.005,
            "mu_gain": 1.0,
            "kla_heads": 4,
            "kla_state_dim": 16,
            "kla_num_layers": 1,
            "kla_hidden_dim": 64,
            "kla_slow_stride": 8,
            "lookback": 64,  # 256→64; 5.3h lookback; features (dist_ma_128 etc.) handle longer context
            "horizon": 6,
            "max_horizon": 24,
            "target_horizon_fee_coverage": 1.5,
            "min_required_horizon_fee_coverage": 0.3,
            "prob_mu_blend": 0.3,
            "prob_temperature": 1.0,
            "min_sigma_for_prob": 0.003,
            "min_pred_variance": 1e-8,
            "max_pred_variance": 2.0,
            "direction_epsilon": 5e-5,
            "balance_direction_loss": True,
            "direction_pos_weight_min": 0.5,
            "direction_pos_weight_max": 3.0,  # was 6.0; caps class reweighting to avoid FLAT-memorization
            "cls_logit_l2": 1e-4,
            "mu_l2_reg": 0.0,
            "seed": 42,
            "net_target": False,  # gross directional labels: ~48% UP / 4% FLAT / 48% DOWN instead of 23/55/22; simpler task
            "learning_rate": 1e-4,  # was 3e-4; slower avoids overfitting on val
            "min_val_directional_accuracy": 0.505,  # 0.5% above random; 0.52 was too strict vs market
            "min_baseline_improvement": 0.005,  # must beat best baseline by 0.5%; 0.02 was unrealistic
            "brier_improvement_margin": 0.0,  # was -0.05; must match or beat baseline Brier
            "collapse_max_class_share": 0.90,
            "collapse_min_side_share": 0.10,  # was 0.03; prevents FLAT-collapse local minima
            "early_stopping_patience": 10,  # increased: allow deeper pattern learning
            "run_simple_baselines": True,
            "baseline_epochs": 5,
            "cls_loss_weight": 1.5,  # was 6.0; 6.0 caused FLAT weight=36× (mem overfitting)
            "nll_loss_clip": -8.0,  # financial returns: optimal NLL ≈ 0.5*log(σ²) ≈ -7 → -2.0 killed var_head  # -3.0→-2.0; prevents sigma collapsing below exp(-2)≈0.14
            "label_smoothing": 0.1,  # CE memorization prevention; 0.1 is standard
            "sequence_stride": 4,   # sample every 4th window; reduces 99.6% overlap to ~94%; ~21k independent sequences
            "weight_decay": 0.05,  # explicit L2 reg; AdamW default 0.01 is too weak for this model
            "head_dropout": 0.4,  # was 0.3; aggressive dropout for small capacity model
            "mu_bias_strength": 1.0,
            "training_start_str": "365 days ago UTC",  # was 180 days; full year = all market regimes
            "training_kline_interval": "5m",
            "training_max_bars": 110000,  # 365d × 288 bars/day = 105,120; this cap leaves headroom
            "training_bar_seconds": 60,
            "training_min_rows": 10000,
            "batch_size": 128,
        },
        risk_limits={
            "max_notional": 500.0,
            "max_abs_position_qty": 0.01,
            "allow_short_selling": True,
            "estimated_fee_bps": 5.0,
            "use_score_sizing": True,
            "score_to_qty_scale": 0.0005,
            "min_score_for_size": 0.0,
            "strength_mode": "alpha_score",
            "same_side_min_edge_improvement": 0.01,
            "max_same_side_scale_in": 0,
            "weak_reentry_edge_threshold": 0.08,
            "shadow_predictions_required": 0,
        },
        strategy={
            "threshold": 0.51,
            "min_edge": 0.001,
            "min_confidence": 0.51,
            "min_expected_value": 0.0,
            "max_sigma": 1.0,
            "min_cooldown_seconds": 0.0,
            "min_bars_between_signals": 6,
            "flip_extra_entry": 0.001,
            "allow_scale_in": False,
            "allow_short_entries": True,
            "enforce_mu_prob_agreement": False,
            "enable_neutral_exit": True,
            "neutral_exit_fraction": 0.75,
            "neutral_exit_respects_cooldown": True,
            "max_holding_seconds": 7200.0,
            "allowed_horizons": [],
            "min_confirm_score": -0.2,
            "stop_loss_bps": 150.0,
            "stop_loss_sigma_scale": 1.0,
            "stop_loss_sigma_ref": 0.001,
        },
        simulation={"initial_cash": 10000.0},
        web={
            "host": "0.0.0.0",
            "port": int(os.getenv("APP_PORT", "8080")),
            "hold_open_seconds": 120,
            "event_delay_seconds": 0.01,
            "live_min_interval_seconds": 0.35,
            "live_stream_type": "kline_5m",
            "live_kline_interval": "5m",
            "train_downsample_seconds": 300.0,
            "fallback_rest_seconds": 300,
            "fallback_poll_interval_seconds": 2.0,
            "warmup_start_str": "3 days ago UTC",
            "warmup_kline_interval": "5m",
            "warmup_min_points": 256,
            "demo_replay_bar_seconds": 0.8,
            "demo_replay_bars": 400,
        },
    )
    state_store = StateStore()
    position_store = PositionStore()
    wallet = SimulationWallet(initial_cash=float(config.simulation.get("initial_cash", 10000.0)))
    wallet.reset()
    time_source = SystemTimeSource()

    # =====================================================
    # 2. INFRASTRUCTURE ADAPTERS
    # =====================================================
    
    bus = AsyncInMemoryEventBus()
    store = EventStoreImpl()

    # =====================================================
    # 3. CORE DOMAIN SERVICES + ML MODELS
    # =====================================================
    
    # KLA feature state builder + KLA predictor
    estimator = KLAStateEstimator()
    predictor = KLAPredictor()

    # News sentiment live state (shared between background poller and risk gate)
    news_state = NewsStateService(
        ema_alpha=0.4,
        strong_threshold=0.45,
        stale_minutes=120.0,
    )
    
    # Strategy & risk policy
    strategy = ThresholdStrategy(threshold=float(config.model.get("signal_threshold", 0.50)))
    policy = RiskPolicy(rules=[
        ConfidenceRule(
            min_confidence=float(
                config.strategy.get(
                    "min_confidence",
                    config.model.get("signal_threshold", 0.50),
                )
            )
        ),
        NewsSentimentGateRule(sentiment_provider=news_state, strong_threshold=0.45),
        SlippageGuardRule(max_slippage_bps=float(config.get("risk_limits.max_slippage_bps", 50.0))),
        MaxQtyRule(max_qty=0.001, min_qty=0.0001),
        MaxNotionalRule(max_notional=float(config.get("risk_limits.max_notional", 1000.0))),
        MaxAbsPositionRule(max_abs_position_qty=float(config.get("risk_limits.max_abs_position_qty", 0.01))),
    ])

    # =====================================================
    # 4. BROKER GATEWAY ADAPTER
    # =====================================================
    
    broker = MockBrokerGateway(
        initial_cash=float(config.simulation.get("initial_cash", 10000.0)),
        bus=bus,
    )
    await broker.connect()

    # =====================================================
    # 5. PIPELINE STAGES (Application layer)
    # =====================================================
    
    state_stage = StateEstimationStage(estimator=estimator, state_store=state_store)
    pred_stage = PredictionStage(predictor=predictor)
    signal_stage = SignalStage(strategy=strategy)
    risk_stage = RiskStage(policy=policy)
    execution_uc = ExecutionUseCase(broker=broker, bus=bus)
    exec_stage = ExecutionStage(execution=execution_uc)
    execution_order_handler = ExecutionOrderEventHandler(
        execution=execution_uc,
        position_store=position_store,
        wallet=wallet,
        bus=bus,
        config=config,
    )

    # =====================================================
    # 6. TRADING ENGINE (Main Facade)
    # =====================================================
    
    engine = TradingEngine(
        bus=bus,
        state_store=state_store,
        position_store=position_store,
        wallet=wallet,
        config=config,
        time_source=time_source,
        state_stage=state_stage,
        pred_stage=pred_stage,
        signal_stage=signal_stage,
        risk_stage=risk_stage,
        exec_stage=exec_stage,
    )

    # =====================================================
    # 7. OFFLINE TRAINING ADAPTER
    # =====================================================
    
    training_engine = TrainingEngine()

    # =====================================================
    # 8. OBSERVABILITY & WEB DASHBOARD
    # =====================================================
    
    perf_tracker = PerformanceTracker(
        initial_equity=wallet.initial_cash,
        peak_equity=wallet.initial_cash,
        risk_free_rate_annual=float(config.model.get("risk_free_rate_annual", 0.0)),
        periods_per_year=int(config.model.get("risk_free_periods_per_year", 252)),
    )
    diagnostics = ModelDiagnosticsService(
        direction_epsilon=float(config.model.get("direction_epsilon", 5e-5))
    )
    metrics = SimpleMetrics()
    tap = EventTapHandler(
        performance_tracker=perf_tracker,
        diagnostics=diagnostics,
        metrics=metrics,
        wallet=wallet,
        position_store=position_store,
    )
    recorder = DataRecorder(store=store, flush_interval_ms=1000)
    metrics_agg = MetricsAggregator()
    
    dashboard = DashboardAPI(
        performance_tracker=perf_tracker,
        metrics_agg=metrics_agg,
        get_model_summary=lambda: {
            **diagnostics.summary(),
            "strategy_diagnostics": strategy.diagnostics(),
            "current_version": predictor.current_version(),
        },
        get_positions_snapshot=position_store.snapshot,
        get_simulation_results=lambda: simulation_snapshot(wallet, position_store, perf_tracker, policy),
        get_wallet_history=lambda limit: wallet.history[-limit:],
        get_health_status=lambda: dashboard_health(config),
        get_risk_diagnostics=policy.diagnostics,
        get_training_summary=lambda: training_state,
    )
    ws_streamer = WebSocketStreamer(performance_tracker=perf_tracker)
    http_server = HttpServer(
        api=dashboard,
        ws=ws_streamer,
        host=str(config.web.get("host", "127.0.0.1")),
        port=int(config.web.get("port", 8000)),
    )

    # =====================================================
    # 9. EVENT BUS SUBSCRIPTIONS
    # =====================================================
    
    bus.subscribe("MarketData", engine)
    bus.subscribe("OrderFilled", execution_order_handler)
    bus.subscribe("OrderRejected", execution_order_handler)
    bus.subscribe("*", tap)  # Observability tap
    bus.subscribe("*", recorder)  # Event store

    # =====================================================
    # 10. DEMO: OFFLINE TRAINING
    # =====================================================

    training_state: dict = {
        "status": "idle",
        "started_at": None,
        "finished_at": None,
        "last_result": None,
        "history": [],
    }

    # Start web server first so training status is visible immediately.
    await http_server.start()
    browser_base = f"http://127.0.0.1:{int(config.web.get('port', 8000))}"
    print(f"Dashboard (browser): {browser_base}/")
    
    print("\n=== OFFLINE TRAINING DEMO ===")

    def downsample_trades_by_interval(df, min_interval_seconds: float):
        if df.empty or min_interval_seconds <= 0:
            return df
        min_interval_ms = int(max(min_interval_seconds, 0.0) * 1000)
        if min_interval_ms <= 0:
            return df

        kept_indices: list[int] = []
        last_ts_ms: int | None = None
        for idx, row in df.iterrows():
            ts_ms = int(getattr(row["T"], "value", 0) // 1_000_000)
            if last_ts_ms is None or (ts_ms - last_ts_ms) >= min_interval_ms:
                kept_indices.append(int(idx))
                last_ts_ms = ts_ms
        if not kept_indices:
            return df.iloc[:1].reset_index(drop=True)
        return df.loc[kept_indices].reset_index(drop=True)

    def kline_interval_to_seconds(interval: str) -> int:
        token = str(interval).strip().lower()
        if token.endswith("s"):
            return max(1, int(token[:-1] or "1"))
        if token.endswith("m"):
            return max(1, int(token[:-1] or "1") * 60)
        if token.endswith("h"):
            return max(1, int(token[:-1] or "1") * 3600)
        if token.endswith("d"):
            return max(1, int(token[:-1] or "1") * 86400)
        return 60

    def estimate_horizon_fee_coverage(prices: list[float], horizon: int, roundtrip_fee_rate: float) -> float:
        if horizon <= 0 or len(prices) <= horizon or roundtrip_fee_rate <= 0:
            return 0.0
        abs_returns: list[float] = []
        for idx in range(0, len(prices) - horizon):
            p_now = float(prices[idx])
            p_fut = float(prices[idx + horizon])
            if p_now <= 0:
                continue
            abs_returns.append(abs((p_fut - p_now) / p_now))
        if not abs_returns:
            return 0.0
        abs_returns.sort()
        median_abs_move = abs_returns[len(abs_returns) // 2]
        return median_abs_move / roundtrip_fee_rate

    def choose_horizon_by_coverage(
        prices: list[float],
        base_horizon: int,
        max_horizon: int,
        target_coverage: float,
        roundtrip_fee_rate: float,
    ) -> tuple[int, float]:
        if not prices:
            return max(1, base_horizon), 0.0

        current = max(1, base_horizon)
        cap = max(current, max_horizon)
        best_h = current
        best_cov = estimate_horizon_fee_coverage(prices, current, roundtrip_fee_rate)

        while current < cap:
            cov = estimate_horizon_fee_coverage(prices, current, roundtrip_fee_rate)
            if cov >= target_coverage:
                return current, cov
            if cov > best_cov:
                best_cov = cov
                best_h = current
            next_h = min(cap, current * 2)
            if next_h == current:
                break
            current = next_h

        final_cov = estimate_horizon_fee_coverage(prices, best_h, roundtrip_fee_rate)
        return best_h, final_cov
    
    # Build training dataset from historical closed klines.
    training_state["status"] = "running"
    training_state["started_at"] = datetime.now(timezone.utc).isoformat()

    training_dataset: list[dict] = []
    training_source = ""
    training_kline_interval = str(
        config.model.get("training_kline_interval", config.web.get("live_kline_interval", "1m"))
    )
    training_interval_seconds = kline_interval_to_seconds(training_kline_interval)
    train_downsample_seconds = float(
        config.web.get("train_downsample_seconds", float(training_interval_seconds))
    )
    training_min_rows = int(config.model.get("training_min_rows", 10000))
    roundtrip_fee_rate = 2.0 * max(float(config.get("risk_limits.estimated_fee_bps", 10.0)), 0.0) / 10_000.0
    try:
        train_df = get_historical_klines_with_taker(
            symbol="BTCUSDT",
            interval=training_kline_interval,
            start_str=str(config.model.get("training_start_str", "3 days ago UTC")),
            max_bars=int(config.model.get("training_max_bars", 200000)),
        )
        if train_df.empty:
            raise ValueError("No historical klines in configured time window")

        if train_downsample_seconds > float(training_interval_seconds):
            train_df = downsample_trades_by_interval(train_df, train_downsample_seconds)

        train_df = train_df.sort_values("T").reset_index(drop=True)
        train_df["ret"] = train_df["p"].pct_change().fillna(0.0)

        if len(train_df) < training_min_rows:
            raise ValueError(
                f"Insufficient training rows from historical klines: {len(train_df)} < {training_min_rows}. "
                "Increase training_start_str window and/or training_max_bars."
            )

        # Fetch funding rates and align to kline timestamps (nearest past settlement)
        print("[TRAIN] Fetching funding rates...")
        funding_rate_map = get_historical_funding_rates(
            symbol="BTCUSDT",
            start_str=str(config.model.get("training_start_str", "3 days ago UTC")),
        )
        print(f"[TRAIN] Funding rate entries fetched: {len(funding_rate_map)}")

        def _nearest_funding(ts_ms: int) -> float:
            if not funding_rate_map:
                return 0.0
            # find most recent funding settlement at or before ts_ms
            best_ts = max((t for t in funding_rate_map if t <= ts_ms), default=None)
            return funding_rate_map[best_ts] if best_ts is not None else 0.0

        price_series = [float(v) for v in train_df["p"].tolist()]
        selected_horizon, estimated_cov = choose_horizon_by_coverage(
            prices=price_series,
            base_horizon=int(config.model.get("horizon", 5)),
            max_horizon=int(config.model.get("max_horizon", 240)),
            target_coverage=float(config.model.get("target_horizon_fee_coverage", 1.5)),
            roundtrip_fee_rate=roundtrip_fee_rate,
        )
        config.model["horizon"] = int(selected_horizon)
        print(
            "[TRAIN] Horizon selection: "
            f"selected={selected_horizon}, estimated_fee_coverage={estimated_cov:.4f}, target={float(config.model.get('target_horizon_fee_coverage', 1.5)):.4f}"
        )
        if estimated_cov < float(config.model.get("min_required_horizon_fee_coverage", 1.0)):
            raise ValueError(
                "Horizon fee coverage too low for live deployment: "
                f"{estimated_cov:.4f} < {float(config.model.get('min_required_horizon_fee_coverage', 1.0)):.4f}."
            )

        training_dataset = [
            {
                "price": float(row["p"]),
                "open":  float(row.get("o", row["p"])),
                "high":  float(row.get("h", row["p"])),
                "low":   float(row.get("l", row["p"])),
                "return": float(row["ret"]),
                "volume": float(row["q"]),
                "ts_ms": int(getattr(row["T"], "value", 0) // 1_000_000),
                "taker_buy_vol": float(row.get("taker_buy_vol", -1.0)),
                "funding_rate": _nearest_funding(int(getattr(row["T"], "value", 0) // 1_000_000)),
            }
            for _, row in train_df.iterrows()
        ]
        config.model["training_bar_seconds"] = int(training_interval_seconds)
        training_source = (
            f"binance-historical-klines-{training_kline_interval}"
            f"-downsampled-{train_downsample_seconds:.3f}s"
        )
    except Exception as error:
        training_state["status"] = "failed"
        training_state["last_result"] = {
            "deployable": False,
            "stage": "data_build",
            "reason": "training-data-build-failed",
            "error": str(error),
            "dataset_size": 0,
            "data_source": "unavailable",
        }
        training_state["history"] = training_engine.get_training_history()
        print(f"[WARN] Real training data build failed, continuing without new model: {error}")
    
    if training_state["status"] != "failed" and training_dataset:
        # Train on historical data (synchronous - no I/O, pure computation)
        try:
            training_result = training_engine.train(
                dataset=training_dataset,
                instrument=Instrument(symbol="BTCUSDT"),
                epochs=40,
                batch_size=int(config.model.get("batch_size", 32)),
                hidden_size=64,
                num_layers=2,
                learning_rate=float(config.model.get("learning_rate", 3e-4)),
                use_nll_loss=True,
                lookback=int(config.model.get("lookback", 64)),
                horizon=int(config.model.get("horizon", 5)),
                kla_heads=int(config.model.get("kla_heads", 4)),
                kla_state_dim=int(config.model.get("kla_state_dim", 32)),
                kla_num_layers=int(config.model.get("kla_num_layers", 3)),
                kla_hidden_dim=int(config.model.get("kla_hidden_dim", 64)),
                kla_slow_stride=int(config.model.get("kla_slow_stride", 12)),
                direction_epsilon=float(config.model.get("direction_epsilon", 5e-5)),
                balance_direction_loss=bool(config.model.get("balance_direction_loss", True)),
                direction_pos_weight_min=float(config.model.get("direction_pos_weight_min", 0.5)),
                direction_pos_weight_max=float(config.model.get("direction_pos_weight_max", 6.0)),
                cls_logit_l2=float(config.model.get("cls_logit_l2", 1e-4)),
                mu_l2_reg=float(config.model.get("mu_l2_reg", 0.0)),
                estimated_fee_bps=float(config.get("risk_limits.estimated_fee_bps", 10.0)),
                net_target=bool(config.model.get("net_target", True)),
                seed=int(config.model.get("seed", 42)),
                early_stopping_patience=int(config.model.get("early_stopping_patience", 5)),
                min_val_directional_accuracy=float(config.model.get("min_val_directional_accuracy", 0.52)),
                min_baseline_improvement=float(config.model.get("min_baseline_improvement", 0.03)),
                brier_improvement_margin=float(config.model.get("brier_improvement_margin", 0.0)),
                collapse_max_class_share=float(config.model.get("collapse_max_class_share", 0.90)),
                collapse_min_side_share=float(config.model.get("collapse_min_side_share", 0.05)),
                run_simple_baselines=bool(config.model.get("run_simple_baselines", True)),
                baseline_epochs=int(config.model.get("baseline_epochs", 2)),
                cls_loss_weight=float(config.model.get("cls_loss_weight", 2.0)),
                nll_loss_clip=float(config.model.get("nll_loss_clip", -10.0)),
                head_dropout=float(config.model.get("head_dropout", 0.0)),
                weight_decay=float(config.model.get("weight_decay", 0.01)),
                label_smoothing=float(config.model.get("label_smoothing", 0.0)),
                sequence_stride=int(config.model.get("sequence_stride", 1)),
            )
            training_result["data_source"] = training_source
            training_result["dataset_size"] = len(training_dataset)
            training_result["train_downsample_seconds"] = train_downsample_seconds
            training_state["last_result"] = training_result
            training_state["history"] = training_engine.get_training_history()
            training_state["status"] = "completed"
            print(f"Training result: {training_result}")
        except Exception as error:
            training_state["status"] = "failed"
            training_state["last_result"] = {
                "deployable": False,
                "stage": "training",
                "reason": "offline-training-failed",
                "error": str(error),
                "dataset_size": len(training_dataset),
                "data_source": training_source,
            }
            training_state["history"] = training_engine.get_training_history()
            print(f"[WARN] Offline training failed, continuing with current model: {error}")
    elif training_state["status"] != "failed":
        training_state["status"] = "skipped"
        training_state["last_result"] = {
            "deployable": False,
            "stage": "training",
            "reason": "training-skipped-empty-dataset",
            "dataset_size": 0,
            "data_source": training_source or "unavailable",
        }
        training_state["history"] = training_engine.get_training_history()
        print("[TRAIN] Offline training skipped: empty dataset.")

    training_state["finished_at"] = datetime.now(timezone.utc).isoformat()
    
    # Push trained model only if validation gate approves deployment
    if bool(training_state.get("last_result", {}).get("deployable", False)):
        await training_engine.push_model(predictor)
        print(f"Model pushed. Current version: {predictor.current_version()}")
    else:
        deploy_skip_reason = (
            training_state.get("last_result", {}).get("deploy_reason")
            or training_state.get("last_result", {}).get("reason")
            or training_state.get("last_result", {}).get("error")
            or "validation-gate-not-satisfied"
        )
        print(f"[TRAIN] Deployment skipped: {deploy_skip_reason}")

    # =====================================================
    # 11. START SERVERS
    # =====================================================
    
    await bus.start()
    print(f"\n=== SIMULATION STARTED ===")
    print(f"initial_cash={wallet.initial_cash}")
    print(f"WebAPI (browser): {browser_base}/simulation")
    print(f"Dashboard (browser): {browser_base}/")
    print(f"Health (browser): {browser_base}/health")

    # =====================================================
    # 12. OPTIONAL: INITIAL BATCH LOAD (for warmup)
    # =====================================================
    
    # Optional: Load recent closed klines for warmup before WebSocket
    try:
        print("\n[WARMUP] Loading recent closed klines for initial state...")
        warmup_lookback = int(config.model.get("lookback", 64))
        warmup_min_points = max(
            int(config.web.get("warmup_min_points", warmup_lookback + 8)),
            warmup_lookback + 8,
        )
        warmup_kline_interval = str(
            config.web.get("warmup_kline_interval", config.web.get("live_kline_interval", "1m"))
        )
        warmup_interval_seconds = kline_interval_to_seconds(warmup_kline_interval)
        warmup_df = get_historical_klines_time_window(
            symbol="BTCUSDT",
            interval=warmup_kline_interval,
            start_str=str(config.web.get("warmup_start_str", "3 days ago UTC")),
            max_bars=max(warmup_min_points * 4, warmup_min_points),
        )
        if warmup_df.empty:
            raise ValueError("No historical klines for warmup")
        warmup_df = warmup_df.sort_values("T").reset_index(drop=True)
        warmup_downsample_seconds = float(
            config.web.get("train_downsample_seconds", float(warmup_interval_seconds))
        )
        if warmup_downsample_seconds > float(warmup_interval_seconds):
            warmup_df = downsample_trades_by_interval(warmup_df, warmup_downsample_seconds)
        if len(warmup_df) > warmup_min_points:
            warmup_df = warmup_df.tail(warmup_min_points).reset_index(drop=True)
        points = to_price_points(warmup_df)
        warmup_count = 0
        for warmup_event in iter_price_points_to_market_events(points, symbol="BTCUSDT", correlation_id="warmup"):
            await bus.publish(warmup_event)
            warmup_count += 1

        print(f"[WARMUP] Processed {warmup_count} historical closed klines")
        if warmup_count < warmup_lookback:
            print(
                f"[WARMUP][WARN] Processed points ({warmup_count}) < lookback ({warmup_lookback}). "
                "Initial predictions may be unstable."
            )
    except Exception as error:
        print(f"[WARN] Warmup failed, starting fresh: {error}")

    # =====================================================
    # 13. RSS + SENTIMENT NEWS (one-time load + background polling)
    # =====================================================

    async def _poll_news_loop() -> None:
        poll_interval = int(config.web.get("news_poll_interval_seconds", 300))
        while True:
            await asyncio.sleep(poll_interval)
            try:
                df = add_sentiment_columns(get_all_news())
                news_state.update_from_df(df)
                diag = news_state.diagnostics()
                print(
                    f"[NEWS] Sentiment updated: score={diag['sentiment_current']:.3f} "
                    f"({diag['sentiment_label']}), articles={diag['recent_article_count']}"
                )
                news_count = 0
                for news_event in iter_news_df_to_events(df.head(20), correlation_prefix="live-news"):
                    await bus.publish(news_event)
                    news_count += 1
            except Exception as error:
                print(f"[NEWS] Poll failed: {error}")

    try:
        news_df = add_sentiment_columns(get_all_news())
        news_state.update_from_df(news_df)
        diag = news_state.diagnostics()
        print(
            f"[NEWS] Initial sentiment: score={diag['sentiment_current']:.3f} "
            f"({diag['sentiment_label']}), articles={diag['recent_article_count']}"
        )
        news_count = 0
        for news_event in iter_news_df_to_events(news_df.head(50), correlation_prefix="demo-news"):
            await bus.publish(news_event)
            news_count += 1
        print(f"News events published: {news_count}")
    except Exception as error:
        print(f"[WARN] RSS/sentiment processing failed: {error}")

    asyncio.create_task(_poll_news_loop())

    # =====================================================
    # 13b. DEMO REPLAY (gyors visszajátszás historikus adatokból)
    # =====================================================
    demo_speed = float(config.web.get("demo_replay_bar_seconds", 0.0))
    if demo_speed > 0 and training_dataset:
        # Demo módban: fali idő alapú cooldown ki (a visszajátszás mesterséges tempójú),
        # de bar-szám alapú cooldown (min_bars_between_signals) marad — ritka jelek.
        config.strategy["min_cooldown_seconds"] = 0.0
        config.strategy["min_bars_between_signals"] = 10  # ~10 perc demo adatnál
        demo_bars_count = int(config.web.get("demo_replay_bars", 400))
        warmup_skip = int(config.web.get("warmup_min_points", 256))
        demo_end = max(0, len(training_dataset) - warmup_skip)
        demo_start = max(0, demo_end - demo_bars_count)
        demo_slice = training_dataset[demo_start:demo_end]
        print(f"\n=== DEMO VISSZAJÁTSZÁS === ({len(demo_slice)} bar, {demo_speed:.1f}s/bar)")
        try:
            for idx, entry in enumerate(demo_slice):
                await bus.publish(MarketDataEvent(
                    payload={
                        "symbol": "BTCUSDT",
                        "price": float(entry["price"]),
                        "o": float(entry.get("open", entry.get("o", entry["price"]))),
                        "h": float(entry.get("high", entry.get("h", entry["price"]))),
                        "l": float(entry.get("low",  entry.get("l", entry["price"]))),
                        "qty": float(entry["volume"]),
                        "timestamp": int(entry["ts_ms"]),
                    },
                    source="demo-replay",
                    correlation_id=f"demo-{entry['ts_ms']}",
                ))
                if (idx + 1) % 50 == 0:
                    sim_snap = wallet.snapshot(position_store)
                    print(f"[DEMO] {idx + 1}/{len(demo_slice)} bar | equity={sim_snap.get('equity', 0):.2f}")
                await asyncio.sleep(demo_speed)
            print(f"[DEMO] Visszajátszás kész. Equity: {wallet.snapshot(position_store).get('equity', 0):.2f}")
        except KeyboardInterrupt:
            print("\n[DEMO] Megszakítva")
            raise

    # =====================================================
    # 14. WEBSOCKET LIVE STREAM (REAL-TIME TRADING)
    # =====================================================
    
    live_stream_type = str(config.web.get("live_stream_type", "kline_1m"))
    live_kline_interval = str(config.web.get("live_kline_interval", "1m"))
    print("\n=== LIVE WEBSOCKET STREAM STARTED ===")
    print("Real-time Binance data streaming. Stop with CTRL+C")
    print(f"Using {live_stream_type} stream and publishing only closed candles")
    
    # WebSocket stream handler (closed kline candles)
    ws_stream = BinanceWebSocketStream(
        symbol="BTCUSDT",
        stream_type=live_stream_type,
        max_reconnects=20,
        max_queue_size=int(config.web.get("ws_internal_queue_size", 5000)),
        websocket_max_queue=config.web.get("ws_transport_max_queue", None),
    )
    total_published = 0
    total_dropped = 0
    last_closed_candle_ts_ms: int | None = None
    last_rest_key: tuple[int, float, float] | None = None
    
    async def handle_websocket_message(msg: dict) -> None:
        """Process each WebSocket message and publish to event bus."""
        nonlocal total_published, total_dropped, last_closed_candle_ts_ms
        
        # Parse kline message and ignore interim candle updates.
        kline_data = parse_kline_message(msg)
        if not kline_data:
            return
        if not bool(kline_data.get("is_closed", False)):
            return

        close_ts_ms = int(kline_data.get("time", 0))
        if last_closed_candle_ts_ms is not None and close_ts_ms <= last_closed_candle_ts_ms:
            total_dropped += 1
            return
        if close_ts_ms <= 0:
            total_dropped += 1
            return
        last_closed_candle_ts_ms = close_ts_ms

        # Create market data event
        market_event = MarketDataEvent(
            payload={
                "symbol": str(kline_data.get("symbol", "BTCUSDT")),
                "price": float(kline_data.get("price", 0.0)),
                "o": float(kline_data.get("open", kline_data.get("price", 0.0))),
                "h": float(kline_data.get("high", kline_data.get("price", 0.0))),
                "l": float(kline_data.get("low",  kline_data.get("price", 0.0))),
                "qty": float(kline_data.get("volume", 0.0)),
                "timestamp": close_ts_ms,
            },
            source="binance-websocket-kline",
            correlation_id=f"ws-kline-{close_ts_ms}",
        )
        
        # Publish to event bus
        await bus.publish(market_event)
        total_published += 1
        
        # Dashboard snapshot every 10 closed candles
        if total_published % 10 == 0:
            print(f"[LIVE] Processed {total_published} closed candles | Price: {float(kline_data.get('price', 0.0)):.2f}")
            print(f"[LIVE] Dropped {total_dropped} websocket updates (dedup/invalid)")
            print(f"[DASHBOARD] Equity: {dashboard.get_performance().get('equity', 0):.2f}")

    async def run_rest_fallback_loop() -> None:
        """Fallback data source when websocket is unstable/unreachable."""
        nonlocal total_published, total_dropped, last_closed_candle_ts_ms, last_rest_key
        fallback_seconds = int(config.web.get("fallback_rest_seconds", 0))
        if fallback_seconds <= 0:
            return

        poll_interval = max(float(config.web.get("fallback_poll_interval_seconds", 2.0)), 0.2)
        end_ts = asyncio.get_running_loop().time() + fallback_seconds
        print(f"\n[FALLBACK] WebSocket unavailable. Switching to REST polling for {fallback_seconds}s")

        while asyncio.get_running_loop().time() < end_ts:
            try:
                df = get_historical_klines_with_taker(
                    symbol="BTCUSDT",
                    interval=live_kline_interval,
                    start_str=str(config.web.get("fallback_start_str", "30 minutes ago UTC")),
                    max_bars=5,
                )
                if not df.empty:
                    df = df.sort_values("T").reset_index(drop=True)
                    row = df.iloc[-1]
                    ts_ms = int(getattr(row["T"], "value", 0) // 1_000_000)
                    price = float(row["p"])
                    qty = float(row["q"])
                    key = (ts_ms, price, qty)
                    if key == last_rest_key:
                        total_dropped += 1
                    elif last_closed_candle_ts_ms is not None and ts_ms <= last_closed_candle_ts_ms:
                        total_dropped += 1
                    else:
                        await bus.publish(
                            MarketDataEvent(
                                payload={
                                    "symbol": "BTCUSDT",
                                    "price": price,
                                    "o": float(row.get("o", price)),
                                    "h": float(row.get("h", price)),
                                    "l": float(row.get("l", price)),
                                    "qty": qty,
                                    "timestamp": ts_ms,
                                },
                                source="binance-rest-kline-fallback",
                                correlation_id=f"rest-kline-{ts_ms}",
                            )
                        )
                        total_published += 1
                        last_closed_candle_ts_ms = ts_ms
                        last_rest_key = key
            except Exception as error:
                print(f"[FALLBACK] Poll failed: {error}")
            await asyncio.sleep(poll_interval)
    
    user_interrupted = False
    try:
        # Start WebSocket stream (blocks until stopped or max reconnects exceeded)
        await ws_stream.start(handle_websocket_message)
        
        # If we reach here, stream stopped (max reconnects or error)
        print(f"\n[STREAM ENDED] Processed {total_published} total closed candles")
        await run_rest_fallback_loop()
        
    except KeyboardInterrupt:
        user_interrupted = True
        print("\n[SHUTDOWN] User interrupted (CTRL+C)")
    except Exception as fatal_error:
        print(f"\n[FATAL ERROR] {fatal_error}")
        import traceback
        traceback.print_exc()
        await run_rest_fallback_loop()
    finally:
        # =====================================================
        # 15. GRACEFUL SHUTDOWN & FINAL REPORT
        # =====================================================
        
        await ws_stream.stop()
        await asyncio.sleep(0.25)
        await bus.stop()
        await recorder.flush()
        await store.append(Event(
            event_type="RunFinished",
            payload={"count": total_published},
            source="demo"
        ))

        print("\n=== FINAL DASHBOARD SNAPSHOT ===")
        print("Performance:", dashboard.get_performance())
        print("Model Summary:", dashboard.get_model_summary())
        print("Positions:", dashboard.get_positions())
        print("Simulation:", dashboard.get_simulation_results())
        print(f"\nTraining History: {training_engine.get_training_history()}")
        print(f"Current Model Version: {predictor.current_version()}")
        
        await http_server.stop()


if __name__ == "__main__":
    asyncio.run(run_demo())
