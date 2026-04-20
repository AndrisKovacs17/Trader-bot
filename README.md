# KLA-Mamba kereskedési rendszer — Felhasználói útmutató

Ez a dokumentum a [diplomamunka](diplomamunka/main.pdf) kiegészítő felhasználói
leírása. A fejlesztői dokumentáció a dolgozatban (4–5. fejezet) található.

## 1. Mire való a szoftver?

A program egy kriptovaluta-kereskedési rendszer demóalkalmazása, amely a
Binance tőzsde nyilvános adatait használja. A rendszer:

* Valós idejű piaci adatokat vesz át a Binance WebSocket API-n keresztül,
* Egy saját fejlesztésű Kalman-szűrővel bővített Mamba (KCMamba) modellel
  rövidtávú ár-előrejelzést készít,
* Kockázati szabályokkal szűrt, paper trading üzemmódú kereskedési
  szignálokat generál,
* Webes irányítópulton (dashboard) megjeleníti a modell állapotát, a
  szignálokat, az egyenleget és a diagnosztikákat.

A program **nem** éles kereskedésre készült; a pénzügyi veszteségek
elkerülése érdekében minden order a beépített szimulációs brókerbe fut.

## 2. Célközönség

| Felhasználó típus | Tipikus feladat |
|---|---|
| Kutató / hallgató | A KCMamba modell teljesítményének reprodukciója a `tests/` suite-on keresztül |
| Fejlesztő | A hexagonális architektúra bővítése új adapterekkel |
| Szakmai bíráló | A rendszer működésének megtekintése a dashboardon |

## 3. Minimális rendszerkövetelmény

| Komponens | Minimum | Javasolt |
|---|---|---|
| Operációs rendszer | Linux (Ubuntu 22.04+) / Windows 10+ / macOS 13+ | Linux |
| Python | 3.11 | 3.12 |
| CPU | 4 mag, x86-64 | 8 mag |
| RAM | 4 GB | 8 GB |
| GPU | — (CPU-n is fut) | NVIDIA CUDA 12.x kompatibilis |
| Lemez | 2 GB (dataset cache-sel együtt ~150 MB) | 5 GB |
| Hálózat | Binance REST/WS (kimenő HTTPS, 443) | stabil szélessáv |

## 4. Telepítés

```bash
# 1. repository klónozása
git clone https://github.com/AndrisKovacs17/Trader-bot.git
cd Trader-bot

# 2. virtuális környezet létrehozása
python3.12 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 3. függőségek telepítése
pip install -r requirements.txt     # futtatáshoz
pip install -r requirements-ml.txt  # a benchmark/tréning scriptekhez
```

A `torch` GPU-s változatához az NVIDIA által ajánlott wheelt kell használni
(<https://pytorch.org/get-started/>). CPU-n is minden futtatható, csak a
benchmark lassabb.

## 5. A program indítása

### 5.1 Fő demó (live + dashboard)

```bash
python main.py
```

Ez elindít:

* egy háttérben futó tréningfolyamatot (offline dataset alapján),
* a Binance WebSocket élő adatfolyamot,
* a webes dashboardot a `http://127.0.0.1:8000/` címen.

Leállítás: **Ctrl+C**. A program ilyenkor szabályosan lezárja a WebSocketet,
kiüríti az eseménypuffert, leállítja a HTTP-szervert és kiírja a végleges
pozíció-összegzést.

### 5.2 Tesztek futtatása

```bash
python -m pytest tests/ -q
```

Várt kimenet: `71 passed`.

## 6. Dashboard — felhasználói felület

A dashboard magyar nyelvű, a felső navigációs sávban kategorizált menükkel:

| Menü | Oldal | Tartalom |
|---|---|---|
| Pénzügyi | Áttekintés (`/`) | Egyenleg, P&L, aktív pozíciók, élő chart |
| Pénzügyi | Modell (`/model.html`) | Modellverzió, kalibráció, diagnosztikai metrikák |
| Pénzügyi | Jelek (`/signals.html`) | Generált szignálok listája, kockázati státusszal |
| Pénzügyi | Tréning (`/training.html`) | Offline tréningfolyamat előrehaladása |
| Benchmark | Szintetikus teszt (`/benchmark.html`) | Kontrollált zajszintű szintetikus teszt eredménye |
| Benchmark | LSTM / Mamba összehasonlítás (`/ltsf_benchmark.html`) | LTSF adaton mért eredmények |
| KLA elemzés | Kalman-szűrő diagnosztika (`/kla_kalman.html`) | K, A, R mátrixok időbeli alakulása |
| KLA elemzés | Modell komplexitás térkép (`/complexity.html`) | Paraméterszám × teljesítmény |

**Súgó a dashboardon**: a jobb alsó sarokban megjelenő `?` gombra kattintva
oldal-specifikus súgóablak nyílik, amely elmagyarázza az adott képernyő
funkcióit.

## 7. Tipikus hibaüzenetek és teendők

| Üzenet | Ok | Teendő |
|---|---|---|
| `[WARN] RSS/sentiment processing failed: ...` | RSS forrás átmenetileg elérhetetlen | Figyelmen kívül hagyható; a program hír nélkül folytatja |
| `[FALLBACK] WebSocket unavailable. Switching to REST polling` | A WebSocket kapcsolat instabil | A program automatikusan áttér REST-polling fallback módra; manuális beavatkozás nem szükséges |
| `[FATAL ERROR] ...` + traceback | Váratlan belső hiba | A program lezárja a kapcsolatokat és megpróbál REST fallbacket; ha ismétlődik, nyisson hibajegyet |
| `[SHUTDOWN] User interrupted (CTRL+C)` | Szabályos megszakítás | Nincs teendő; a `finally` ág eltakarít |
| `Bad request: API key required` | Érvénytelen Binance kulcs | Csak historikus/WebSocket adatokhoz nem kell kulcs; éles rendeléshez sem, mert a rendszer paper trading módban fut |
| `OSError: [Errno 98] Address already in use` | A 8000-es port foglalt | Állítsa le a foglaló folyamatot, vagy módosítsa a `config.web.port` értéket |

A program minden hibaüzenetet a `stderr`-re ír, a rendes kimenet a `stdout`-ra
megy — így `nohup python main.py > run.log 2> run.err &` módon külön
naplózható.

## 8. Biztonsági megfontolások

* A rendszer paper trading módban fut; nem indít valós tőzsdei pozíciót.
* A Binance kulcsok (ha mégis használ) csak olvasási jogosultságúak legyenek.
* A dashboard csak a lokális interfészen (`127.0.0.1`) hallgat, nem nyilvános.
* A tréning cache (`adapters/offline_training/dataset_cache/`) nincs a
  repositoryban (lásd `.gitignore`).

## 9. Megszakíthatóság

A program hosszú műveletei (tréning, benchmark sweep, live stream) **Ctrl+C**-vel
bármikor biztonságosan leállíthatók. A shutdown-ág garantálja, hogy:

* a WebSocket kapcsolat lezárul,
* az in-memory esemény-pufferek flush-olódnak a tárolóba,
* a HTTP-szerver lezárja az aktív kapcsolatokat,
* a végleges állapot kiírásra kerül a konzolra.

## 10. Architektúra rövid összefoglaló

A rendszer **hexagonális architektúrát** (Ports & Adapters) követ:

* `core/domain` — tartomány-entitások, értékobjektumok (Event, Signal, Order)
* `core/application` — használati esetek, portok, pipeline szakaszok
* `core/ml` — ML-portok és alap-prediktor
* `adapters/infrastructure` — Binance, RSS hírek, WebSocket, REST
* `adapters/web` — HTTP/WebSocket dashboard
* `adapters/offline_training` — batch tréning és benchmark
* `adapters/testing` — backtest és mock bróker

A részletes tervezési dokumentáció a dolgozat 4. fejezetében található
(`diplomamunka/main.pdf`).

---

**Verzió**: 1.0 (2026-04-19) · **Licenc**: az ELTE szakdolgozatra vonatkozó szabályzat szerint
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
