# Diplomamunka – Async Clean/Hexagonal Trading Architecture (Fully Hexagonal)

High-integrity trading system implementing **Hexagonal Architecture** with AsyncIO.

## Architecture Overview

### Core Layer (Domain-Driven Design)

- **`core/domain`**: Domain entities, value objects, aggregates
  - `events.py`: Base Event + type-specific events (MarketData, Order, Signal, Risk, Model)
  - `models.py`: Instrument, Signal, Order, Fill, RiskResult, Position
  - `strategy.py`: IStrategy interface (trading signal generation)
  - `risk.py`: IRiskRule, RiskPolicy (risk evaluation via Chain of Responsibility)

- **`core/ml`**: ML domain services with model lifecycle
  - `services.py`: 
    - `IStateEstimator` port - state estimation (Kalman, Simple variants)
    - `IPredictor` port - price prediction
    - **`IModelUpdatePort`** - Training Engine → Predictor updates
    - **`IModelLifecycle`** - Staging → Active model application
    - `MambaPredictor` - Dual-model (active + staging) implementation

- **`core/ops`**: Operations metrics contract
- **`core/analytics`**: Analytics services (performance tracking, diagnostics)

### Application Layer (Use Cases + Orchestration)

- **`core/application/ports`**: Hexagonal boundary (interfaces for adapters)
  - `IEventBusPort` - Event publication/subscription
  - `IEventHandlerPort` - Event processing
  - `IStateRepository` - State persistence
  - `IBrokerGatewayPort` - Order execution
  - `ITimeSource` - Time provider
  - `IExecutionUseCase` - Order submission/handling

- **`core/application/stages`**: Pipeline orchestration
  - `IMarketDataStage`, `IStateEstimationStage`, `IPredictionStage`
  - `ISignalStage`, `IRiskStage`, `IExecutionStage`

- **`core/application/engine`**: TradingEngine facade
  - Orchestrates full pipeline: Market → State → Prediction → Signal → Risk → Execution
  - Integrates **IModelLifecycle** for ML model updates

### Adapter Layer (Infrastructure Implementation)

- **`adapters/infrastructure`**: Core adapters
  - `event_bus.py`: AsyncInMemoryEventBus, InMemoryEventStore
  - `execution.py`: ExecutionUseCase, SimpleBrokerGateway
  - `binance_feed.py`: Real market data from Binance API
  - `news_feed.py`: RSS news + sentiment analysis

- **`adapters/offline_training`**: **Training Engine** (NEW)
  - `training_engine.py`: Offline model training with zero-downtime deployment
  - `train(dataset, epochs)` - Train models
  - `push_model(update_port)` - Deploy via staging model pattern

- **`adapters/web`**: Dashboard API
  - REST + WebSocket real-time streaming

- **`adapters/testing`**: Backtest framework

## Key Features

### Fully Hexagonal (Zero Infrastructure Coupling)

- All ports defined in `core/application/ports` + `core/ml/services`
- Adapters implement ports
- Domain has no dependencies outside of itself

### ML Model Lifecycle (NEW)

```
TrainingEngine (Offline)
  └─ train(dataset) → export_weights() → push_model(predictor)
       └─ IModelUpdatePort
            ├─ Staging model receives weights
            └─ apply_pending_update() → Active swap (zero-downtime)
```

### Trading Pipeline

```
MarketData → State → Prediction → Signal → Risk → Execution
```

## Installation

```bash
pip install pandas feedparser nltk python-binance
```

## Running

```bash
python main.py
```

Output includes:
- Offline training demo
- Live simulation with Binance data
- Dashboard at `http://127.0.0.1:8000/`

## Configuration

```python
config = Config(
    symbols=["BTCUSDT"],
    model={"use_kalman": True, "signal_threshold": 0.50},
    simulation={"initial_cash": 10000.0},
)
```

---

**Status**: MVP (Production-ready core, scalable adapter layer)  
**Date**: 2026-02-22
