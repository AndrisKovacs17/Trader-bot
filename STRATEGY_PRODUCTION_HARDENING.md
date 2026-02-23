# Strategy Layer Production Hardening - Detailed Implementation

## 🎯 Overview
Ez a dokumentum a `ThresholdStrategy` production-grade finomítását írja le, amely 6 haladó szintű kockázati pontot kezel.

---

## 📋 Implementált Javítások

### 1️⃣ **Thread-Safe Internal State Protection**

#### ❌ Korábbi Probléma
```python
# Nem védett mutable state:
_last_signal_ts_by_symbol: dict[str, datetime]
_last_signal_bar_by_symbol: dict[str, int]

# Race condition multi-threaded környezetben:
def _in_cooldown(self, symbol: str, ...) -> bool:
    last_ts = self._last_signal_ts_by_symbol.get(symbol)  # ❌ Race!
    ...

# Signal emit után:
self._last_signal_ts_by_symbol[symbol] = now  # ❌ Race!
```

**Miért veszélyes?**
- Ha több worker thread párhuzamosan hív `on_prediction()`-t ugyanarra a symbolra:
  - Thread A: read + write közben Thread B is read-el → inconsistent state
  - Counter elveszhet vagy duplikálódhat
  - Cooldown check hamisan pozitív/negatív lehet

#### ✅ Megoldás
```python
from threading import Lock

@dataclass(slots=True)
class ThresholdStrategy:
    ...
    _state_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    
    def _in_cooldown(self, symbol: str, ...) -> bool:
        with self._state_lock:
            last_ts = self._last_signal_ts_by_symbol.get(symbol)
            last_bar = self._last_signal_bar_by_symbol.get(symbol)
        # Check logic itt már thread-safe másolatokkal
        ...
    
    def on_prediction(self, pred, ctx):
        ...
        with self._state_lock:
            self._last_signal_ts_by_symbol[symbol] = now
            self._last_signal_bar_by_symbol[symbol] = bar_count
```

**Működés:**
- `_state_lock: Lock` minden strategy instance-nek saját lock-ja van
- `with self._state_lock:` biztosítja, hogy egyidejűleg csak 1 thread férjen hozzá a dict-ekhez
- **Single-threaded engine:** Lock overhead elhanyagolható (~30-50 ns)
- **Multi-threaded engine:** Megelőzi a race condition-öket, konzisztens cooldown tracking

**Tesztelés:**
```python
def test_thread_safe_cooldown_tracking():
    # ThreadPoolExecutor 10 worker
    # Ugyanaz a strategy, ugyanaz a symbol
    # Első emit után cooldown alatt
    # Mind a 10 concurrent call-nak visszautasítást kell kapnia
    assert all(r is False for r in results)  # ✅ Passed
```

---

### 2️⃣ **Explicit Bar Index Contract**

#### ❌ Korábbi Probléma
```python
@staticmethod
def _bar_count(ctx: RunContext) -> int:
    if hasattr(ctx, "bar_index"):  # ❌ Implicit contract
        return int(ctx.bar_index)
    return int(ctx.config.get("strategy.bar_count", len(ctx.wallet.history)))
```

**Miért gyenge?**
- `hasattr()` runtime check, nem compile-time contract
- Ha RunContext-nek nincs `bar_index` attribútuma → fallback-re esik
- Wallet implementation detail (`len(history)`) szivárog be a stratégiába
- Determinizmus veszélyes: fallback más eredményt adhat replay során

#### ✅ Megoldás
```python
@staticmethod
def _bar_count(ctx: RunContext) -> int:
    # Explicit contract: RunContext MUST have bar_index attribute.
    # Fallback csak backward compatibility-hez; production enginenek
    # explicit bar_index-et kell biztosítania.
    if not hasattr(ctx, "bar_index"):
        # Implicit fallback, de logolható warning production-ben
        return int(ctx.config.get("strategy.bar_count", len(ctx.wallet.history)))
    return int(ctx.bar_index)
```

**Továbbfejlesztési irány:**
```python
# Opció A: RunContext interface-ben explicit property
@dataclass(slots=True)
class RunContext:
    bar_index: int = 0  # ✅ Explicit field, nem implicit

# Opció B: Assertion production módban
if not hasattr(ctx, "bar_index"):
    logger.warning("RunContext missing bar_index, using fallback (non-deterministic)")
    # vagy strict módban:
    raise ValueError("RunContext MUST have bar_index for deterministic replay")
```

**Engine felelőssége:**
```python
# engine.py
def _build_context(self, ...) -> RunContext:
    ctx = RunContext(
        ...
        bar_index=len(self.wallet.history),  # ✅ Explicit assignment
    )
```

**Előnyök:**
- Determinisztikus replay: `bar_index` mindig ugyanaz adott pozícióban
- Wallet-független: strategy nem függ wallet implementációtól
- Tesztelhető: explicit `bar_index` override test fixture-ökben

---

### 3️⃣ **Edge-Based Signal Score Option**

#### ❌ Korábbi Limitáció
```python
# Egyetlen signal quality metric:
signal_score = abs(pred.mu) * confidence
# confidence = max(prob_up, 1.0 - prob_up)

# Problém:
# - prob_up=0.9 → confidence=0.9 (magas)
# - prob_up=0.51 → confidence=0.51 (alacsony)
# De edge figyelmen kívül van hagyva!
```

**Mi az edge?**
- `edge = abs(prob_up - 0.5)`
- Mennyire "határozott" a directional conviction
- prob_up=0.9 → edge=0.4 (erős)
- prob_up=0.51 → edge=0.01 (gyenge, de confidence=0.51)

#### ✅ Megoldás: Választható Scoring Mode
```python
@dataclass(slots=True)
class ThresholdStrategy:
    use_edge_score: bool = False  # False: |mu|*confidence, True: edge*|mu|
    ...

def on_prediction(self, pred, ctx):
    ...
    edge = abs(prob_up - 0.5)
    
    # Konfigurálható scoring:
    use_edge_score = bool(ctx.config.get("strategy.use_edge_score", self.use_edge_score))
    if use_edge_score:
        signal_score = edge * abs(float(pred.mu))  # Directional conviction
    else:
        signal_score = abs(float(pred.mu)) * confidence  # Volatility-neutral quality
```

**Mikor melyiket használd:**

| Mode | Formula | Use Case |
|------|---------|----------|
| `use_edge_score=False` (default) | `\|mu\| × confidence` | Binary model output, volatility-neutral quality proxy |
| `use_edge_score=True` | `edge × \|mu\|` | Directional models, explicit boundary conviction weighted |

**Példa:**
```python
# prob_up=0.75, mu=0.02, sigma=0.01
edge = 0.25
confidence = 0.75

# Default mode:
signal_score = 0.02 * 0.75 = 0.015

# Edge mode:
signal_score = 0.25 * 0.02 = 0.005

# Ha min_signal_score=0.01 → default pass, edge mode block
```

**Tesztelés:**
```python
def test_use_edge_score_mode():
    ctx.config.strategy["use_edge_score"] = True
    ctx.config.strategy["min_signal_score"] = 0.005
    
    # Edge = 0.25, mu = 0.02 → score = 0.005 (pass)
    signal = strategy.on_prediction(pred_edge, ctx)
    assert signal is not None
    
    # Edge = 0.05, mu = 0.02 → score = 0.001 (block)
    assert strategy.on_prediction(pred_low_edge, ctx) is None  # ✅ Passed
```

---

### 4️⃣ **Min Sigma Protection Against Extreme Strength**

#### ❌ Korábbi Probléma
```python
vol = max(float(pred.sigma), 1e-8)  # ❌ Hardcoded
risk_scaled_strength = abs(float(pred.mu)) / vol

# Ha sigma → 0:
# mu=0.02, sigma=1e-10 → strength = 0.02 / 1e-10 = 2e8 !!!
# → Extrém order size
# → Risk policy-ra van bízva, de stratégia szinten is védeni kell
```

**Miért veszélyes?**
- Model output noise: sigma prediction nem tökéletes
- Outlier protection: egyetlen rossz sigma érték gigantikus pozíciót okozhat
- Defense-in-depth: stratégia + risk + execution mind védjen

#### ✅ Megoldás: Konfigurálható Min Sigma
```python
@dataclass(slots=True)
class ThresholdStrategy:
    min_sigma: float = 1e-8  # Default, de konfiguráció override-olhatja
    ...

def on_prediction(self, pred, ctx):
    ...
    min_sigma = float(ctx.config.get("strategy.min_sigma", self.min_sigma))
    
    # Volatility clamping: prevent extreme strength from tiny sigma.
    vol = max(float(pred.sigma), min_sigma)
    risk_scaled_strength = abs(float(pred.mu)) / vol
```

**Tuning Guide:**
```yaml
strategy:
  min_sigma: 0.001  # Crypto: magasabb volatilitás
  min_sigma: 0.0001 # FX: alacsonyabb volatilitás
  min_sigma: 1e-8   # Konzervatív (csak division-by-zero védelem)
```

**Trade-off:**
- **Túl alacsony min_sigma:** Outlier-ek átjutnak, extrém strength
- **Túl magas min_sigma:** Valóban alacsony volatilitás periódusok alul-sized

**Recommended:** Backtest-tel kalibráld a `min_sigma`-t az asset class-ra!

**Tesztelés:**
```python
def test_min_sigma_prevents_extreme_strength():
    ctx.config.strategy["min_sigma"] = 0.001
    pred_tiny_sigma = make_pred(prob_up=0.8, mu=0.02, sigma=1e-10)
    signal = strategy.on_prediction(pred_tiny_sigma, ctx)
    
    # Strength = 0.02 / 0.001 / 1 = 20.0 (not 2e8!)
    assert signal.strength > 0.0
    assert signal.strength < 1000.0  # ✅ Bounded
```

---

### 5️⃣ **Max Strength Cap for Safety**

#### ❌ Korábbi Probléma
```python
strength = max(0.0, horizon_scaled_strength)
# Nincs felső korlát!

# Ha mu=0.5, sigma=0.01, horizon=1:
# strength = 0.5 / 0.01 / 1 = 50.0 → 50 BTC order?!
```

**Miért probléma?**
- Model outlier-ek: Ha predikció hibás, hatalmas veszteség
- Risk layer-re bízva: `max_qty` véd, de layered defense jobb
- Explicit intent: "soha ne lépjem túl X strength-et stratégia szinten"

#### ✅ Megoldás: Optional Max Strength Cap
```python
@dataclass(slots=True)
class ThresholdStrategy:
    max_strength: float = 0.0  # 0.0=disabled; >0 caps strength
    ...

def on_prediction(self, pred, ctx):
    ...
    max_strength = float(ctx.config.get("strategy.max_strength", self.max_strength))
    
    # Base strength application with optional upper cap.
    strength = max(0.0, horizon_scaled_strength)
    if max_strength > 0.0:
        strength = min(strength, max_strength)
```

**Használati mód:**
```yaml
strategy:
  max_strength: 10.0  # Soha ne haladja meg a 10.0-t stratégia szinten
  # Risk layer max_qty=1.0 továbbra is véd, de ez explicit strategy cap

  max_strength: 0.0   # Disabled (default), risk layer-re bízva
```

**Defense-in-Depth Filozófia:**

```
Prediction → Strategy → Risk → Execution
              ↓           ↓        ↓
           max_strength  max_qty  order size limit
             (opcionális) (kötelező) (exchange limit)
```

**Tesztelés:**
```python
def test_max_strength_caps_output():
    ctx.config.strategy["max_strength"] = 5.0
    pred_high = make_pred(prob_up=0.9, mu=0.5, sigma=0.01)
    # Természetes strength >> 5.0
    signal = strategy.on_prediction(pred_high, ctx)
    assert signal.strength <= 5.0  # ✅ Capped
```

---

### 6️⃣ **Cooldown Reset Contract Documentation**

#### ❌ Korábbi Hiány
```python
def reset(self) -> None:
    """Reset strategy internal state for deterministic replay boundaries."""
    self._last_signal_ts_by_symbol = {}
    self._last_signal_bar_by_symbol = {}
```

**Probléma:**
- Nem dokumentált, hogy KI és MIKOR kell hívni
- Engine fejlesztő nem tudja, hogy replay előtt reset() kell
- Multi-session replay esetén cooldown állapot "átszivárog"

#### ✅ Megoldás: Comprehensive Documentation
```python
def reset(self) -> None:
    """Reset strategy internal state for deterministic replay boundaries.
    
    CRITICAL: Engine MUST call this at replay session start to clear cooldown state.
    
    Multi-threading:
    - Thread-safe: uses _state_lock to protect dictionary mutations.
    - Single-threaded engine: lock overhead minimal.
    - Multi-threaded engine: prevents race conditions during concurrent on_prediction calls.
    
    Replay contract:
    - Call reset() before replaying historical bars.
    - Ensures cooldown tracking starts fresh.
    - Does NOT reset config-driven parameters (threshold, min_edge, etc.).
    """
    with self._state_lock:
        self._last_signal_ts_by_symbol = {}
        self._last_signal_bar_by_symbol = {}
```

**Engine Integration Example:**
```python
# backtest.py
class BacktestEngine:
    def replay_session(self, start_date: datetime, end_date: datetime):
        # ✅ CRITICAL: Reset strategy cooldown state
        self.strategy.reset()
        
        for bar in self._load_bars(start_date, end_date):
            ctx = self._build_context(bar)
            pred = self.model.predict(ctx.state)
            signal = self.strategy.on_prediction(pred, ctx)
            ...
```

**Multi-Session Replay Example:**
```python
# Test multiple scenarios
for scenario in ["bull_market", "bear_market", "sideways"]:
    strategy.reset()  # ✅ Fresh start for each scenario
    for bar in load_scenario(scenario):
        ...
```

**Thread-Safety Note:**
- `reset()` is thread-safe: uses `_state_lock`
- Ha engine multi-threaded: ne hívj `reset()`-et miközben `on_prediction()` fut
- Best practice: reset() csak initialization fázisban, nem hot path-ban

---

## 🧪 Test Coverage

### Új Tesztek (4 db)
1. **`test_min_sigma_prevents_extreme_strength`**
   - Tiny sigma (1e-10) → clamped to min_sigma (0.001)
   - Strength bounded, nem explode to infinity
   
2. **`test_max_strength_caps_output`**
   - High mu/low sigma → natural strength >> 5.0
   - Config `max_strength=5.0` → signal.strength <= 5.0
   
3. **`test_use_edge_score_mode`**
   - `use_edge_score=True` → edge-based filtering
   - High edge pass, low edge block (same confidence scenario)
   
4. **`test_thread_safe_cooldown_tracking`**
   - ThreadPoolExecutor 10 workers
   - Concurrent `on_prediction()` calls during cooldown
   - All blocked correctly, no race condition

### Összes Teszt: **26/26 PASSED** ✅

---

## 📊 Configuration Reference

### Teljes Strategy Config
```yaml
strategy:
  # Threshold & Quality Gates
  threshold: 0.55           # Directional decision boundary
  min_edge: 0.02            # Minimum edge over 50%
  min_confidence: 0.55      # Minimum max(prob_up, 1-prob_up)
  min_signal_score: 0.0     # Quality score floor
  
  # Scoring Mode
  use_edge_score: false     # false: |mu|*conf, true: edge*|mu|
  
  # Volatility Protection
  max_sigma: 1.0            # Upper bound on prediction uncertainty
  min_sigma: 1e-8           # Lower bound (division-by-zero guard)
  
  # Strength Limits
  max_strength: 0.0         # 0.0=disabled; >0 caps final strength
  
  # Cooldown
  min_cooldown_seconds: 0.0 # Time-based signal throttle
  min_bars_between_signals: 0  # Bar-based signal throttle
  
  # Horizon
  allowed_horizons: []      # Empty=all; [1,2,4]=filter
  horizon_scale_mode: "inverse"  # "inverse" or "none"
  
  # Position Handling
  flip_extra_entry: 0.0     # Extra qty when flipping position
```

### Recommended Production Values
```yaml
# Conservative Crypto
strategy:
  threshold: 0.60
  min_edge: 0.05
  min_confidence: 0.65
  min_signal_score: 0.01
  use_edge_score: false
  max_sigma: 0.10
  min_sigma: 0.001
  max_strength: 10.0
  min_cooldown_seconds: 300.0
  min_bars_between_signals: 5
  allowed_horizons: [1, 2]
  horizon_scale_mode: "inverse"
  flip_extra_entry: 0.5
```

---

## 🎓 Production Deployment Checklist

- [x] **Thread-safety:** `_state_lock` protects mutable state
- [x] **Explicit contracts:** `bar_index` fallback documented
- [x] **Configurable scoring:** `use_edge_score` for model flexibility
- [x] **Outlier protection:** `min_sigma` prevents extreme strength
- [x] **Safety caps:** `max_strength` layered defense
- [x] **Replay support:** `reset()` contract documented
- [x] **Test coverage:** 26/26 tests passing (13 strategy + 13 risk)
- [x] **Config defaults:** All new parameters in `Config` dataclass

---

## 🔧 Maintenance Notes

### Ha új worker pool-t adsz hozzá:
- ✅ Strategy thread-safe, nincs teendő

### Ha új bar_index forrást adsz hozzá:
- ✅ Update `engine._build_context()` to populate `ctx.bar_index`
- ⚠️  Fallback megtartva backward compatibility-hez

### Ha új scoring mode-ot akarsz:
1. Add új mode a `use_edge_score` IF-ELSE láncba
2. Add config default-ot `Config.strategy`
3. Írj explicit tesztet az új mode-ra

### Ha cooldown logikát változtatsz:
- ⚠️  Update `reset()` docstring if contract changes
- ⚠️  Ensure `_state_lock` protects new mutable state

---

## 📚 Related Documentation
- [Domain Models](./DOMAIN_MODELS_EXPLANATION.md) - Signal, Order, Fill, Position
- [Risk Policy](./RISK_PRODUCTION_HARDENING.md) - Risk layer deep dive
- [Replay Guide](./REPLAY_DETERMINISM.md) - Deterministic replay best practices

---

**Version:** 1.0  
**Date:** 2026-02-22  
**Status:** Production-Ready ✅
