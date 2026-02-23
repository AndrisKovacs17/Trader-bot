from __future__ import annotations

import asyncio
import logging

# Configure logging to see pipeline debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

from adapters.infrastructure.binance_feed import (
    BinanceWebSocketStream,
    get_latest_trades,
    get_recent_aggregate_trades,
    parse_trade_message,
    to_price_points,
)
from adapters.infrastructure.event_bus import AsyncInMemoryEventBus, DataRecorder, EventStoreImpl
from adapters.infrastructure.event_mapping import iter_news_df_to_events, iter_price_points_to_market_events
from adapters.infrastructure.execution import (
    ExecutionOrderEventHandler,
)
from adapters.testing.broker import MockBrokerGateway
from adapters.infrastructure.news_feed import add_sentiment_columns, get_all_news
from adapters.infrastructure.time_source import SystemTimeSource
from adapters.observability.event_tap import EventTapHandler
from adapters.offline_training.training_engine import TrainingEngine
from adapters.web.dashboard import DashboardAPI, HttpServer, WebSocketStreamer
from core.analytics.services import MetricsAggregator, ModelDiagnosticsService, PerformanceTracker
from core.application.engine import TradingEngine
from core.application.execution import ExecutionUseCase
from core.application.stages import ExecutionStage, MarketDataStage, PredictionStage, RiskStage, SignalStage, StateEstimationStage
from core.application.stores import Config, PositionStore, SimulationWallet, StateStore
from core.domain.events import Event, ModelLifecycleAppliedEvent
from core.domain.models import Instrument
from core.domain.risk import ConfidenceRule, MaxAbsPositionRule, MaxNotionalRule, MaxQtyRule, RiskPolicy, SlippageGuardRule
from core.domain.strategy import ThresholdStrategy
from core.domain.events import MarketDataEvent
from core.ml.services import KalmanStateEstimator, MambaPredictor, SimpleStateEstimator, ToggleableStateEstimator
from core.ops.contracts import SimpleMetrics


# =====================================================
# HEXAGONAL WIRING: Core + Infrastructure + Adapters
# =====================================================

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
            "use_kalman": True,
            "signal_threshold": 0.50,
            "default_signal": False,
            "default_prob_up": 0.56,
            "default_mu": 0.0005,
            "default_sigma": 0.01,
            "mamba_window": 128,
            "horizon": 1,
        },
        risk_limits={
            "max_notional": 500.0,
            "max_abs_position_qty": 0.01,
        },
        strategy={
            "threshold": 0.55,
            "min_edge": 0.02,
            "min_confidence": 0.55,
            "min_expected_value": 0.0,
            "max_sigma": 1.0,
            "min_cooldown_seconds": 0.0,
            "min_bars_between_signals": 10,
            "allowed_horizons": [],
        },
        simulation={"initial_cash": 10000.0},
        web={"host": "0.0.0.0", "port": 8000, "hold_open_seconds": 120, "event_delay_seconds": 0.01},
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
    
    # State estimation service
    use_kalman = bool(config.model.get("use_kalman", True))
    toggleable_estimator = ToggleableStateEstimator(
        kalman=KalmanStateEstimator(),
        simple=SimpleStateEstimator(),
        use_kalman=use_kalman,
    )
    estimator = toggleable_estimator
    
    # Predictor with model lifecycle (staging -> active)
    predictor = MambaPredictor()
    
    # Strategy & risk policy
    strategy = ThresholdStrategy(threshold=float(config.model.get("signal_threshold", 0.50)))
    policy = RiskPolicy(rules=[
        ConfidenceRule(min_confidence=0.50),
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

    # =====================================================
    # 5. PIPELINE STAGES (Application layer)
    # =====================================================
    
    state_stage = StateEstimationStage(estimator=estimator, state_store=state_store)
    market_stage = MarketDataStage()
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
        market_stage=market_stage,
        state_stage=state_stage,
        pred_stage=pred_stage,
        signal_stage=signal_stage,
        risk_stage=risk_stage,
        exec_stage=exec_stage,
        model_lifecycle=predictor,  # MambaPredictor implements IModelLifecycle
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
    diagnostics = ModelDiagnosticsService()
    metrics = SimpleMetrics()
    tap = EventTapHandler(performance_tracker=perf_tracker, diagnostics=diagnostics, metrics=metrics)
    recorder = DataRecorder(store=store, flush_interval_ms=1000)
    metrics_agg = MetricsAggregator()
    
    dashboard = DashboardAPI(
        performance_tracker=perf_tracker,
        metrics_agg=metrics_agg,
        diagnostics=diagnostics,
        position_store=position_store,
        wallet=wallet,
        config=config,
        set_kalman_enabled=toggleable_estimator.set_use_kalman,
        get_kalman_enabled=toggleable_estimator.get_use_kalman,
        get_risk_diagnostics=policy.diagnostics,
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
    bus.subscribe("ModelLifecycleApplied", engine)
    bus.subscribe("OrderFilled", execution_order_handler)
    bus.subscribe("OrderRejected", execution_order_handler)
    bus.subscribe("*", tap)  # Observability tap
    bus.subscribe("*", recorder)  # Event store

    # =====================================================
    # 10. DEMO: OFFLINE TRAINING
    # =====================================================
    
    print("\n=== OFFLINE TRAINING DEMO ===")
    
    # Create synthetic training dataset
    synthetic_dataset = [
        {"price": 100.0 + i*0.1, "return": 0.001*i, "volume": 1000}
        for i in range(100)
    ]
    
    # Train on historical data (synchronous - no I/O, pure computation)
    training_result = training_engine.train(
        dataset=synthetic_dataset,
        instrument=Instrument(symbol="BTCUSDT"),
        epochs=2,
        batch_size=32
    )
    print(f"Training result: {training_result}")
    
    # Push trained model to production (non-blocking via staging model)
    await training_engine.push_model(predictor)
    print(f"Model pushed. Pending update: {predictor.has_pending_update()}")

    # Trigger dedicated lifecycle event so staged model is applied before live run.
    await bus.publish(
        ModelLifecycleAppliedEvent(
            payload={"reason": "post_training_bootstrap"},
            source="main",
            correlation_id="bootstrap-model-lifecycle",
        )
    )

    # =====================================================
    # 11. START SERVERS
    # =====================================================
    
    await bus.start()
    await http_server.start()
    print(f"\n=== SIMULATION STARTED ===")
    print(f"use_kalman={use_kalman}, initial_cash={wallet.initial_cash}")
    browser_base = f"http://127.0.0.1:{int(config.web.get('port', 8000))}"
    print(f"WebAPI (browser): {browser_base}/simulation")
    print(f"Dashboard (browser): {browser_base}/")
    print(f"Health (browser): {browser_base}/health")

    # =====================================================
    # 12. OPTIONAL: INITIAL BATCH LOAD (for warmup)
    # =====================================================
    
    # Optional: Load recent trades for warmup before WebSocket
    try:
        print("\n[WARMUP] Loading recent trades for initial state...")
        trades_df = get_latest_trades(symbol="BTCUSDT", limit=50)
        points = to_price_points(trades_df)
        warmup_count = 0
        for warmup_event in iter_price_points_to_market_events(points, symbol="BTCUSDT", correlation_id="warmup"):
            await bus.publish(warmup_event)
            warmup_count += 1

        print(f"[WARMUP] Processed {warmup_count} historical trades")
    except Exception as error:
        print(f"[WARN] Warmup failed, starting fresh: {error}")

    # =====================================================
    # 13. RSS + SENTIMENT NEWS (one-time load)
    # =====================================================
    
    try:
        news_df = add_sentiment_columns(get_all_news())
        news_count = 0
        for news_event in iter_news_df_to_events(news_df.head(50), correlation_prefix="demo-news"):
            await bus.publish(news_event)
            news_count += 1
        print(f"News events published: {news_count}")
    except Exception as error:
        print(f"[WARN] RSS/sentiment processing failed: {error}")

    # =====================================================
    # 14. WEBSOCKET LIVE STREAM (REAL-TIME TRADING)
    # =====================================================
    
    print("\n=== LIVE WEBSOCKET STREAM STARTED ===")
    print("Real-time Binance data streaming. Stop with CTRL+C")
    print("Using aggTrade stream (more stable than individual trades)")
    
    # WebSocket stream handler (aggTrade = aggregated trades, more stable)
    ws_stream = BinanceWebSocketStream(
        symbol="BTCUSDT",
        stream_type="aggTrade",
        max_reconnects=20,
        max_queue_size=int(config.web.get("ws_internal_queue_size", 5000)),
        websocket_max_queue=config.web.get("ws_transport_max_queue", None),
    )
    total_published = 0
    total_dropped = 0
    min_emit_interval = float(config.web.get("live_min_interval_seconds", 0.35))
    last_emit_ts = 0.0
    last_trade_id: int | None = None
    
    async def handle_websocket_message(msg: dict) -> None:
        """Process each WebSocket message and publish to event bus."""
        nonlocal total_published, total_dropped, last_emit_ts, last_trade_id
        
        # Parse trade message
        trade_data = parse_trade_message(msg)
        if not trade_data:
            return

        # De-duplicate reconnect overlap on Binance trade id
        trade_id = int(trade_data.get("trade_id", 0))
        if last_trade_id is not None and trade_id <= last_trade_id:
            total_dropped += 1
            return

        # Throttle event rate so pipeline can keep up
        now_ts = asyncio.get_running_loop().time()
        if now_ts - last_emit_ts < min_emit_interval:
            total_dropped += 1
            return

        last_emit_ts = now_ts
        last_trade_id = trade_id

        # Create market data event
        market_event = MarketDataEvent(
            payload={
                "symbol": trade_data["symbol"],
                "price": trade_data["price"],
                "qty": trade_data["qty"],
                "timestamp": trade_data["time"],
            },
            source="binance-websocket",
            correlation_id=f"ws-trade-{trade_data['trade_id']}",
        )
        
        # Publish to event bus
        await bus.publish(market_event)
        total_published += 1
        
        # Dashboard snapshot every 50 events
        if total_published % 50 == 0:
            print(f"[LIVE] Processed {total_published} trades | Price: {trade_data['price']:.2f}")
            print(f"[LIVE] Dropped {total_dropped} trades (dedup/throttle)")
            print(f"[DASHBOARD] Equity: {dashboard.get_performance().get('equity', 0):.2f}")
    
    try:
        # Start WebSocket stream (blocks until stopped or max reconnects exceeded)
        await ws_stream.start(handle_websocket_message)
        
        # If we reach here, stream stopped (max reconnects or error)
        print(f"\n[STREAM ENDED] Processed {total_published} total trades")
        
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] User interrupted (CTRL+C)")
    except Exception as fatal_error:
        print(f"\n[FATAL ERROR] {fatal_error}")
        import traceback
        traceback.print_exc()
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
