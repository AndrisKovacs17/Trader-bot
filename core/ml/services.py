from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Protocol, TYPE_CHECKING

from core.domain.events import MarketDataEvent

try:
    import torch
except Exception:
    torch = None  # type: ignore[assignment]

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except Exception:
    selective_scan_fn = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from core.application.stores import RunContext
    from core.domain.models import Instrument


# =====================================================
# VALUE OBJECTS
# =====================================================

# Állapotbecslés value object.
@dataclass(slots=True)
class EstimatedState:
    x: list[float]
    P: list[list[float]]
    features: dict[str, Any]
    confidence: float
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Predikció value object.
@dataclass(slots=True)
class Prediction:
    mu: float
    sigma: float
    prob_up: float
    regime: str
    horizon: int
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =====================================================
# DOMAIN SERVICE PORTS (Hexagonal boundary for ML)
# =====================================================

# State estimator port.
class IStateEstimator(Protocol):
    async def update(self, md: MarketDataEvent, ctx: RunContext) -> EstimatedState:
        ...

    def reset(self, instrument: Instrument) -> None:
        ...

    def get_state(self, instrument: Instrument) -> EstimatedState:
        ...


# Predictor port.
class IPredictor(Protocol):
    async def predict(self, state: EstimatedState, ctx: RunContext) -> Prediction:
        ...

    def current_version(self) -> str:
        ...


# Model update port: Training Engine -> MambaPredictor kommunikáció
class IModelUpdatePort(Protocol):
    async def request_model_update(self, weights: bytes, version: str) -> None:
        ...


# Model lifecycle port: Staging -> Active model átváltás
class IModelLifecycle(Protocol):
    async def apply_pending_update(self) -> dict[str, Any] | None:
        ...

    def has_pending_update(self) -> bool:
        ...


# =====================================================
# STATE ESTIMATOR IMPLEMENTATIONS
# =====================================================

# Minimal "Kalman-szerű" estimator: csak cache-eli az utolsó árat/returnt.
class KalmanStateEstimator:
    def __init__(self) -> None:
        self.models: dict[str, EstimatedState] = {}

    async def update(self, md: MarketDataEvent, ctx: RunContext) -> EstimatedState:
        price = float(md.payload.get("price", 0.0))
        prev = self.models.get(ctx.instrument.symbol)
        prev_price = prev.features.get("price", price) if prev else price
        ret = 0.0 if prev_price == 0 else (price - prev_price) / prev_price
        state = EstimatedState(
            x=[price, ret],
            P=[[1.0, 0.0], [0.0, 1.0]],
            features={"price": price, "return": ret},
            confidence=0.7,
        )
        self.models[ctx.instrument.symbol] = state
        return state

    def reset(self, instrument: Instrument) -> None:
        symbol = getattr(instrument, "symbol", "")
        self.models.pop(symbol, None)

    def get_state(self, instrument: Instrument) -> EstimatedState:
        symbol = getattr(instrument, "symbol", "")
        return self.models[symbol]


# Kalman nélküli egyszerű állapotbecslő (toggle esetére).
class SimpleStateEstimator:
    def __init__(self) -> None:
        self.models: dict[str, EstimatedState] = {}

    async def update(self, md: MarketDataEvent, ctx: RunContext) -> EstimatedState:
        price = float(md.payload.get("price", 0.0))
        prev = self.models.get(ctx.instrument.symbol)
        prev_price = prev.features.get("price", price) if prev else price
        ret = 0.0 if prev_price == 0 else (price - prev_price) / prev_price
        state = EstimatedState(
            x=[price],
            P=[[1.0]],
            features={"price": price, "return": ret},  # ← FIXED: Now calculates return!
            confidence=1.0,
        )
        self.models[ctx.instrument.symbol] = state
        return state

    def reset(self, instrument: Instrument) -> None:
        symbol = getattr(instrument, "symbol", "")
        self.models.pop(symbol, None)

    def get_state(self, instrument: Instrument) -> EstimatedState:
        symbol = getattr(instrument, "symbol", "")
        return self.models[symbol]


class ToggleableStateEstimator:
    def __init__(self, kalman: KalmanStateEstimator, simple: SimpleStateEstimator, use_kalman: bool = True) -> None:
        self.kalman = kalman
        self.simple = simple
        self.use_kalman = use_kalman

    def set_use_kalman(self, enabled: bool) -> None:
        self.use_kalman = bool(enabled)

    def get_use_kalman(self) -> bool:
        return self.use_kalman

    async def update(self, md: MarketDataEvent, ctx: RunContext) -> EstimatedState:
        if self.use_kalman:
            return await self.kalman.update(md, ctx)
        return await self.simple.update(md, ctx)

    def reset(self, instrument: Instrument) -> None:
        self.kalman.reset(instrument)
        self.simple.reset(instrument)

    def get_state(self, instrument: Instrument) -> EstimatedState:
        if self.use_kalman:
            return self.kalman.get_state(instrument)
        return self.simple.get_state(instrument)


# =====================================================
# PREDICTOR IMPLEMENTATIONS
# =====================================================

# MambaPredictor: Active + Staging model pattern, IModelUpdatePort + IModelLifecycle implementáció
class MambaPredictor(IPredictor, IModelUpdatePort, IModelLifecycle):
    """
    Mamba-based predictor with model versioning and online update capability.
    
    Supports:
    - Active model for inference
    - Staging model for pending updates (from offline training)
    - Zero-downtime model updates via `apply_pending_update()`
    """
    
    def __init__(self, version: str = "mvp-v1") -> None:
        # Active model (used for predictions)
        self.active_version = version
        self.active_weights: dict[str, Any] = {}
        
        # Staging model (pending update from training)
        self.staging_version: str | None = None
        self.staging_weights: dict[str, Any] | None = None
        self.update_pending = False

        self.return_history: dict[str, list[float]] = {}
        self.device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))

    def _predictor_weights(self) -> dict[str, Any]:
        if not self.active_weights:
            return {}
        if isinstance(self.active_weights, dict) and isinstance(self.active_weights.get("predictor"), dict):
            return self.active_weights["predictor"]
        if isinstance(self.active_weights, dict):
            return self.active_weights
        return {}

    def _run_selective_scan(self, history: list[float], weights: dict[str, Any]) -> tuple[float, float] | None:
        if torch is None or selective_scan_fn is None or not history:
            return None

        mamba = weights.get("mamba") if isinstance(weights.get("mamba"), dict) else {}
        d_model = int(mamba.get("d_model", 4))
        n_state = int(mamba.get("n_state", 8))
        delta_scale = float(mamba.get("delta_scale", 0.1))

        seq = history[-max(8, min(len(history), 64)):]
        seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
        if seq_tensor.numel() < 2:
            return None

        diff = torch.diff(seq_tensor, prepend=seq_tensor[:1])
        abs_seq = torch.abs(seq_tensor)
        sign_seq = torch.sign(seq_tensor)
        channels = [seq_tensor, diff, abs_seq, sign_seq]
        while len(channels) < d_model:
            channels.append(seq_tensor)
        u = torch.stack(channels[:d_model], dim=0).unsqueeze(0)  # [1, D, L]
        delta = torch.ones_like(u) * delta_scale

        A = torch.tensor(
            mamba.get("A", [[-0.1 for _ in range(n_state)] for _ in range(d_model)]),
            dtype=torch.float32,
            device=self.device,
        )
        B_base = torch.tensor(
            mamba.get("B_base", [0.1 for _ in range(n_state)]),
            dtype=torch.float32,
            device=self.device,
        )
        C_base = torch.tensor(
            mamba.get("C_base", [0.1 for _ in range(n_state)]),
            dtype=torch.float32,
            device=self.device,
        )
        D = torch.tensor(
            mamba.get("D", [0.0 for _ in range(d_model)]),
            dtype=torch.float32,
            device=self.device,
        )
        B = B_base.view(1, -1, 1).expand(1, n_state, u.shape[-1])
        C = C_base.view(1, -1, 1).expand(1, n_state, u.shape[-1])

        out = selective_scan_fn(u, delta, A, B, C, D=D, delta_softplus=True)
        summary = out[:, :, -1].squeeze(0)  # [D]
        sigma = max(1e-6, float(out.std().detach().cpu().item()))

        head = weights.get("head", {}) if isinstance(weights.get("head"), dict) else {}
        w = head.get("w", [])
        b = float(head.get("b", 0.0))
        if isinstance(w, list) and len(w) >= d_model:
            w_tensor = torch.tensor(w[:d_model], dtype=torch.float32, device=self.device)
            mu = float((summary * w_tensor).sum().detach().cpu().item() + b)
        else:
            mu = float(summary.mean().detach().cpu().item())

        return mu, sigma

    async def predict(self, state: EstimatedState, ctx: RunContext) -> Prediction:
        """Predict using active model weights."""
        ret = float(state.features.get("return", 0.0))
        symbol = getattr(ctx.instrument, "symbol", "UNKNOWN")
        history = self.return_history.setdefault(symbol, [])
        history.append(ret)
        if len(history) > int(ctx.config.get("model.mamba_window", 128)):
            del history[:-int(ctx.config.get("model.mamba_window", 128))]

        predictor_weights = self._predictor_weights()

        # Trained selective-scan path (preferred when mamba-ssm is available)
        if predictor_weights:
            try:
                scan_result = self._run_selective_scan(history, predictor_weights)
                if scan_result is not None:
                    mu, sigma = scan_result
                    prob_up = self._sigmoid(mu / (sigma + 1e-8))
                    return Prediction(
                        mu=mu,
                        sigma=sigma,
                        prob_up=min(0.99, max(0.01, prob_up)),
                        regime="mamba-ssm",
                        horizon=int(ctx.config.get("model.horizon", 1)),
                    )
            except Exception:
                # Safe fallback to non-mamba path
                pass

            head = predictor_weights.get("head", {}) if isinstance(predictor_weights.get("head"), dict) else {}
            w = head.get("w", [])
            b = float(head.get("b", 0.0))
            if isinstance(w, list) and w:
                mu = float(w[0]) * ret + b
                sigma = max(1e-6, abs(ret) + float(predictor_weights.get("volatility", 0.01)))
                prob_up = self._sigmoid(mu / sigma)
                return Prediction(
                    mu=mu,
                    sigma=sigma,
                    prob_up=min(0.99, max(0.01, prob_up)),
                    regime="trained-linear",
                    horizon=int(ctx.config.get("model.horizon", 1)),
                )

        # Default signal path for non-model runs (ensures strategy can act)
        if bool(ctx.config.get("model.default_signal", False)):
            base_prob = float(ctx.config.get("model.default_prob_up", 0.60))
            base_prob = min(0.99, max(0.51, base_prob))
            direction = 1.0 if ret >= 0 else -1.0
            prob_up = base_prob if direction >= 0 else (1.0 - base_prob)
            mu = float(ctx.config.get("model.default_mu", 0.001)) * direction
            sigma = float(ctx.config.get("model.default_sigma", 0.01))
            return Prediction(
                mu=mu,
                sigma=sigma,
                prob_up=prob_up,
                regime="default",
                horizon=1,
            )
        
        # Simple heuristic: return előjele alapján
        prob_up = min(0.99, max(0.01, 0.5 + ret * 10))
        
        return Prediction(
            mu=ret,
            sigma=abs(ret) + 0.01,
            prob_up=prob_up,
            regime="normal",
            horizon=1,
        )

    def current_version(self) -> str:
        """Return active model version."""
        return self.active_version

    async def request_model_update(self, weights: bytes, version: str) -> None:
        """
        Training Engine által meghívva: új model súlyok staging-be írása.
        
        Serialization format:
        - Az első 8 byte: version string hossza (uint64)
        - Ezt követően: version string (UTF-8)
        - Maradék: pickled model dict (active_weights)
        """
        import pickle
        
        # Parse version length from first 8 bytes
        if len(weights) < 8:
            raise ValueError("Invalid model update: too short")
        
        version_length = int.from_bytes(weights[:8], byteorder='big')
        if len(weights) < 8 + version_length:
            raise ValueError("Invalid model update: corrupted version field")
        
        parsed_version = weights[8:8+version_length].decode('utf-8')
        if version and parsed_version != version:
            raise ValueError("Model update version mismatch between payload and argument")
        weights_payload = weights[8+version_length:]
        
        # Unpickle weights
        self.staging_weights = pickle.loads(weights_payload)
        self.staging_version = parsed_version
        self.update_pending = True

    async def apply_pending_update(self) -> dict[str, Any] | None:
        """Apply staging model to active (zero-downtime switch)."""
        if not self.update_pending or self.staging_weights is None:
            return None

        old_version = self.active_version
        self.active_weights = self.staging_weights
        self.active_version = self.staging_version or self.active_version
        self.update_pending = False
        self.staging_weights = None
        self.staging_version = None
        return {
            "old_version": old_version,
            "new_version": self.active_version,
            "status": "applied",
        }

    def has_pending_update(self) -> bool:
        """Check if a model update is waiting to be applied."""
        return self.update_pending
