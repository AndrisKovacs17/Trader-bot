from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Protocol, TYPE_CHECKING

from core.domain.events import MarketDataEvent
from core.ml.feature_engineering import build_trade_feature_rows, FEATURE_NAMES

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None

from core.ml.kla_mamba import KLAMambaBlock, KLAMambaStack

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
    confirm_score: float = 0.0  # multi-indicator confirmation: +1=all bullish, -1=all bearish
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


# Model update port: Training Engine -> Predictor kommunikáció
class IModelUpdatePort(Protocol):
    async def request_model_update(self, weights: bytes, version: str) -> None:
        ...


# =====================================================
# STATE ESTIMATOR IMPLEMENTATIONS
# =====================================================

# KLA-hoz szükséges minimális feature state előállító.
class KLAStateEstimator:
    def __init__(self) -> None:
        self.models: dict[str, EstimatedState] = {}

    async def update(self, md: MarketDataEvent, ctx: RunContext) -> EstimatedState:
        price = float(md.payload.get("price", 0.0))
        volume = float(md.payload.get("qty", md.payload.get("volume", 0.0)) or 0.0)
        # Kline WebSocket field "V" = taker buy base volume; trade events don't have it.
        taker_buy_vol = float(md.payload.get("V", md.payload.get("taker_buy_vol", -1.0)) or -1.0)
        raw_ts = md.payload.get("timestamp", None)
        if raw_ts is None:
            event_ts_ms = int(md.ts_event.timestamp() * 1000)
        else:
            event_ts_ms = int(raw_ts or 0)
            if event_ts_ms <= 0:
                event_ts_ms = int(md.ts_event.timestamp() * 1000)
        prev = self.models.get(ctx.instrument.symbol)
        prev_price = prev.features.get("price", price) if prev else price
        ret = 0.0 if prev_price == 0 else (price - prev_price) / prev_price
        # OHLC: available from kline WebSocket streams (fields "o","h","l");
        # falls back to close price for aggTrade/trade streams.
        open_p = float(md.payload.get("open", md.payload.get("o", price)) or price)
        high_p = float(md.payload.get("high", md.payload.get("h", price)) or price)
        low_p  = float(md.payload.get("low",  md.payload.get("l", price)) or price)
        high_p = max(high_p, price, open_p)
        low_p  = min(low_p,  price, open_p)
        state = EstimatedState(
            x=[price, ret, volume],
            P=[[1.0]],
            features={
                "price": price,
                "return": ret,
                "volume": volume,
                "timestamp_ms": event_ts_ms,
                "taker_buy_vol": taker_buy_vol,
                "o": open_p,
                "h": high_p,
                "l": low_p,
            },
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


# =====================================================
# PREDICTOR IMPLEMENTATIONS
# =====================================================

# KLAPredictor: KLA alapú prediktor, közvetlen online modellfrissítéssel
class KLAPredictor(IPredictor, IModelUpdatePort):
    """
    KLA-based predictor with model versioning and online update capability.
    
    Supports:
    - Active model for inference
    - Direct online model updates via `request_model_update()`
    """
    
    def __init__(self, version: str = "mvp-v1") -> None:
        # Active model (used for predictions)
        self.active_version = version
        self.active_model: dict[str, Any] = {}
        
        self.device = "cpu"
        self._compiled_model: Any | None = None
        self._scaler_mean: list[float] | None = None
        self._scaler_std: list[float] | None = None
        self._lookback: int = 1
        self._horizon: int = 1
        self._train_positive_ratio: float = 0.5
        self._train_class_priors: list[float] = [0.33, 0.34, 0.33]
        self._train_mu_mean: float = 0.0
        self._class_count: int = 1
        self._feature_buffers: dict[str, deque[dict[str, float | int]]] = {}

    def _build_runtime_model(self) -> Any | None:
        if torch is None or nn is None:
            return None
        predictor_payload = self.active_model.get("predictor", {}) if isinstance(self.active_model, dict) else {}
        hyper = predictor_payload.get("hyperparams", {}) if isinstance(predictor_payload, dict) else {}
        state_dict = predictor_payload.get("state_dict", {}) if isinstance(predictor_payload, dict) else {}
        scaler = predictor_payload.get("scaler", {}) if isinstance(predictor_payload, dict) else {}
        if not state_dict:
            return None

        model_type = str(hyper.get("model_type", "kla"))
        if model_type != "kla":
            raise ValueError(f"Unsupported model_type for predictor: {model_type}")

        input_size = int(hyper.get("input_size", 3))
        self._class_count = max(1, int(hyper.get("class_count", 1)))
        kla_heads = int(hyper.get("kla_heads", 4))
        kla_state_dim = int(hyper.get("kla_state_dim", 16))
        kla_num_layers = int(hyper.get("kla_num_layers", 1))
        kla_hidden_dim = int(hyper.get("kla_hidden_dim", input_size))
        kla_slow_stride = int(hyper.get("kla_slow_stride", 12))
        self._lookback = max(1, int(hyper.get("lookback", 1)))
        self._horizon = max(1, int(hyper.get("horizon", 1)))
        self._train_positive_ratio = min(max(float(hyper.get("train_positive_ratio", 0.5)), 1e-4), 1.0 - 1e-4)
        priors = hyper.get("train_class_priors", None)
        if isinstance(priors, list) and len(priors) == self._class_count:
            safe_priors = [max(float(x), 1e-6) for x in priors]
            total = sum(safe_priors)
            self._train_class_priors = [x / total for x in safe_priors]
        elif self._class_count == 1:
            self._train_class_priors = [1.0]
        else:
            if self._class_count == 2:
                self._train_class_priors = [1.0 - self._train_positive_ratio, self._train_positive_ratio]
            elif self._class_count == 3:
                remain = max(0.0, 1.0 - self._train_positive_ratio)
                self._train_class_priors = [0.6 * remain, 0.4 * remain, self._train_positive_ratio]
            else:
                self._train_class_priors = [1.0 / float(self._class_count)] * self._class_count
        self._train_mu_mean = float(hyper.get("train_mu_mean", 0.0))

        # Detect architecture from state_dict keys
        _has_kla_stack = any(k.startswith("kla_stack.") for k in state_dict)
        _has_mlp_head = any(k.startswith("mu_head.0.") or k.startswith("up_head.0.") for k in state_dict)
        _head_input = kla_hidden_dim if _has_kla_stack else input_size
        _hh = 64
        if _has_kla_stack:
            backbone = KLAMambaStack(
                feature_dim=input_size,
                hidden_dim=kla_hidden_dim,
                num_layers=kla_num_layers,
                heads=kla_heads,
                d_state=kla_state_dim,
                slow_stride=kla_slow_stride,
            )
            mu_head = nn.Sequential(nn.Linear(_head_input, _hh), nn.SiLU(), nn.Linear(_hh, 1))
            up_head = nn.Sequential(nn.Linear(_head_input, _hh), nn.SiLU(), nn.Linear(_hh, self._class_count))
            var_head = nn.Sequential(nn.Linear(_head_input, _hh), nn.SiLU(), nn.Linear(_hh, 1))
            model = nn.ModuleDict({"kla_stack": backbone, "mu_head": mu_head, "up_head": up_head, "var_head": var_head})
        else:
            backbone = KLAMambaBlock(d_model=input_size, heads=kla_heads, d_state=kla_state_dim)
            if _has_mlp_head:
                mu_head = nn.Sequential(nn.Linear(_head_input, _hh), nn.SiLU(), nn.Linear(_hh, 1))
                up_head = nn.Sequential(nn.Linear(_head_input, _hh), nn.SiLU(), nn.Linear(_hh, self._class_count))
                var_head = nn.Sequential(nn.Linear(_head_input, _hh), nn.SiLU(), nn.Linear(_hh, 1))
            else:
                mu_head = nn.Linear(_head_input, 1)
                up_head = nn.Linear(_head_input, self._class_count)
                var_head = nn.Linear(_head_input, 1)
            model = nn.ModuleDict({"kla_block": backbone, "mu_head": mu_head, "up_head": up_head, "var_head": var_head})
        # Validate checkpoint compatibility — strict=False silently loads partial
        # weights, leaving missing layers at random init without any warning.
        _ckpt_keys = set(state_dict.keys())
        _model_keys = set(model.state_dict().keys())
        _missing_in_ckpt = sorted(_model_keys - _ckpt_keys)
        _unexpected_in_ckpt = sorted(_ckpt_keys - _model_keys)
        _shape_mismatches = [
            (k, tuple(model.state_dict()[k].shape), tuple(state_dict[k].shape))
            for k in _ckpt_keys & _model_keys
            if model.state_dict()[k].shape != state_dict[k].shape
        ]
        if _missing_in_ckpt or _unexpected_in_ckpt or _shape_mismatches:
            raise ValueError(
                f"Saved model artifact is incompatible with current architecture — "
                f"missing={_missing_in_ckpt}, unexpected={_unexpected_in_ckpt}, "
                f"shape_mismatches={_shape_mismatches}. Retrain the model."
            )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(self.device)

        self._scaler_mean = [float(x) for x in scaler.get("mean", [0.0] * input_size)]
        self._scaler_std = [max(float(x), 1e-6) for x in scaler.get("std", [1.0] * input_size)]
        self._feature_buffers = {}
        self._compiled_model = model
        return model

    async def predict(self, state: EstimatedState, ctx: RunContext) -> Prediction:
        """Predict using active model weights."""
        ret = float(state.features.get("return", 0.0))

        if self._compiled_model is None and self.active_model:
            self._build_runtime_model()

        if self._compiled_model is not None and torch is not None:
            price = float(state.features.get("price", 0.0))
            volume = float(state.features.get("volume", 0.0))
            ts_ms = int(state.features.get("timestamp_ms", state.features.get("timestamp", 0)) or 0)
            taker_buy_vol = float(state.features.get("taker_buy_vol", -1.0))
            symbol = getattr(getattr(ctx, "instrument", None), "symbol", "UNKNOWN")
            mean = self._scaler_mean or [0.0]
            std = self._scaler_std or [1.0]

            raw_feature = {
                "price": price,
                "return": ret,
                "volume": volume,
                "ts_ms": ts_ms,
                "taker_buy_vol": taker_buy_vol,
                "o": float(state.features.get("o", price)),
                "h": float(state.features.get("h", price)),
                "l": float(state.features.get("l", price)),
            }
            buffer = self._feature_buffers.get(symbol)
            if buffer is None:
                buffer = deque(maxlen=max(self._lookback * 8, 512))
                self._feature_buffers[symbol] = buffer
            buffer.append(raw_feature)

            engineered_rows, _ = build_trade_feature_rows(list(buffer))
            if not engineered_rows:
                engineered_rows = [[0.0] * len(mean)]

            feature_dim = len(mean)
            def _fit_row(row: list[float]) -> list[float]:
                if len(row) == feature_dim:
                    return row
                if len(row) > feature_dim:
                    return row[:feature_dim]
                return row + ([0.0] * (feature_dim - len(row)))

            engineered_rows = [_fit_row(row) for row in engineered_rows]
            latest_feature = engineered_rows[-1]

            if len(buffer) == 0:
                seq = [latest_feature] * self._lookback
            elif len(engineered_rows) < self._lookback:
                pad = [engineered_rows[0]] * (self._lookback - len(engineered_rows))
                seq = pad + engineered_rows
            else:
                seq = engineered_rows[-self._lookback:]

            normalized_seq = [
                [
                    (row[i] - mean[i]) / std[i]
                    for i in range(feature_dim)
                ]
                for row in seq
            ]
            with torch.no_grad():
                x = torch.tensor([normalized_seq], dtype=torch.float32, device=self.device)
                _kla_key = "kla_stack" if "kla_stack" in self._compiled_model else "kla_block"
                y_seq, _, _, _ = self._compiled_model[_kla_key](x)
                h = y_seq[:, -1, :]
                mu_raw = float(self._compiled_model["mu_head"](h).squeeze().item())
                raw_up_logits = self._compiled_model["up_head"](h)
                prob_temperature = max(float(ctx.config.get("model.prob_temperature", 2.0)), 0.25)

                if self._class_count == 1:
                    prior_debias_strength = min(max(float(ctx.config.get("model.prior_debias_strength", 0.8)), 0.0), 2.0)
                    prior_logit = math.log(self._train_positive_ratio / (1.0 - self._train_positive_ratio))
                    debiased_logit = raw_up_logits - prior_debias_strength * prior_logit
                    prob_up_raw = float(torch.sigmoid(debiased_logit / prob_temperature).squeeze().item())
                else:
                    prior_debias_strength = min(max(float(ctx.config.get("model.prior_debias_strength", 0.8)), 0.0), 2.0)
                    prior = self._train_class_priors
                    if len(prior) != self._class_count:
                        prior = [1.0 / float(self._class_count)] * self._class_count
                    prior_tensor = torch.tensor(prior, dtype=raw_up_logits.dtype, device=raw_up_logits.device)
                    # Clamp before log: near-zero FLAT prior gives log≈-5 → +2.5 constant
                    # boost per 0.5 strength that overwhelms every signal. Min 0.1 keeps
                    # the debias within a reasonable [-2.3, 0] range per class.
                    prior_tensor_safe = torch.clamp(prior_tensor, min=0.1)
                    prior_logits = torch.log(prior_tensor_safe).view(1, -1)
                    debiased_logits = raw_up_logits - prior_debias_strength * prior_logits
                    probs = torch.softmax(debiased_logits / prob_temperature, dim=-1)
                    # Binary (class_count==2): UP=class 1, DOWN=class 0
                    # 3-class (legacy): UP=2, FLAT=1→0.5 mix, DOWN=0
                    if self._class_count >= 3:
                        p_flat = float(probs[:, 1].squeeze().item())
                        p_up = float(probs[:, 2].squeeze().item())
                        prob_up_raw = p_up + 0.5 * p_flat
                    elif self._class_count == 2:
                        prob_up_raw = float(probs[:, 1].squeeze().item())
                    else:
                        prob_up_raw = float(probs[:, -1].squeeze().item())
                if "var_head" in self._compiled_model:
                    variance_raw = float(torch.nn.functional.softplus(self._compiled_model["var_head"](h)).squeeze().item())
                    min_var = max(float(ctx.config.get("model.min_pred_variance", 1e-4)), 1e-8)
                    max_var = max(float(ctx.config.get("model.max_pred_variance", 2.0)), min_var)
                    variance = min(max(variance_raw, min_var), max_var)
                    sigma = max(variance ** 0.5, 1e-4)
                else:
                    sigma = max(abs(mu_raw) * 0.5, 1e-4)
            max_abs_mu = float(ctx.config.get("model.max_abs_mu", 0.02))
            mu_bias_strength = min(max(float(ctx.config.get("model.mu_bias_strength", 1.0)), 0.0), 2.0)
            mu_centered = mu_raw - mu_bias_strength * self._train_mu_mean
            mu_gain = max(float(ctx.config.get("model.mu_gain", 1.0)), 1e-6)
            mu = max(-max_abs_mu, min(max_abs_mu, mu_centered * mu_gain))

            # Blend classifier probability with mu/sigma implied direction to avoid sticky near-0/near-1 outputs.
            min_sigma_for_prob = max(float(ctx.config.get("model.min_sigma_for_prob", 0.03)), 1e-4)
            sigma_ref = max(float(sigma), min_sigma_for_prob)
            mu_implied_prob = 0.5 + 0.5 * math.tanh(mu / sigma_ref)
            prob_mu_blend = min(1.0, max(0.0, float(ctx.config.get("model.prob_mu_blend", 0.35))))
            prob_up = (1.0 - prob_mu_blend) * prob_up_raw + prob_mu_blend * mu_implied_prob
            prob_floor = min(0.49, max(0.0, float(ctx.config.get("model.prob_floor", 0.02))))

            # --- Multi-indicator confirmation score (leading, not lagging) ---
            # Four independent categories:
            #   1. Order flow      : bid_ask_proxy – net taker-buy pressure (leading)
            #   2. Structure       : breakout_32 – position vs 32-bar range
            #   3. Consistency     : trend_quality_8 – fraction of recent bars going same way
            #   4. Conviction      : volume_accel – unusual volume surge
            _fidx: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}

            def _fget(feat_name: str) -> float:
                i = _fidx.get(feat_name, -1)
                return float(latest_feature[i]) if 0 <= i < len(latest_feature) else 0.0

            def _vote(v: float, deadband: float = 0.05) -> float:
                return 1.0 if v > deadband else (-1.0 if v < -deadband else 0.0)

            # 1. Order-flow: bid_ask_proxy ∈ [-1,1] (taker buy surplus is directional)
            _flow_vote = _vote(_fget("bid_ask_proxy"), deadband=0.10)

            # 2. Structural breakout: directional
            _struct_vote = _vote(_fget("breakout_32"), deadband=0.15)

            # 3. Trend consistency: directional
            _quality_vote = _vote(_fget("trend_quality_8"), deadband=0.0)

            # 4. Volume conviction: magnitude only (volume_accel)
            _vol_accel = _fget("volume_accel")
            _vol_conviction = 1.0 if _vol_accel > 0.5 else (0.0 if _vol_accel > -0.5 else -0.5)

            # confirm_score: weighted average of directional votes, boosted by conviction
            confirm_score = (2.0 * _flow_vote + _struct_vote + _quality_vote) / 4.0
            confirm_score = confirm_score * (1.0 + 0.5 * _vol_conviction)
            confirm_score = max(-1.5, min(1.5, confirm_score))

            return Prediction(
                mu=mu,
                sigma=sigma,
                prob_up=min(1.0 - prob_floor, max(prob_floor, prob_up)),
                regime="trained",
                horizon=self._horizon,
                confirm_score=round(confirm_score, 4),
            )

        return Prediction(
            mu=0.0,
            sigma=1.0,
            prob_up=0.5,
            regime="no-model",
            horizon=1,
        )

    def current_version(self) -> str:
        """Return active model version."""
        return self.active_version

    async def request_model_update(self, weights: bytes, version: str) -> None:
        """Training Engine által meghívva: új model súlyok közvetlen aktiválása."""
        import io as _io
        import torch as _torch

        # Parse version length from first 8 bytes
        if len(weights) < 8:
            raise ValueError("Invalid model update: too short")

        version_length = int.from_bytes(weights[:8], byteorder="big")
        if len(weights) < 8 + version_length:
            raise ValueError("Invalid model update: corrupted version field")

        parsed_version = weights[8:8 + version_length].decode("utf-8")
        weights_payload = weights[8 + version_length:]
        # Use torch.load instead of bare pickle.loads to keep deserialisation
        # within PyTorch's own serialisation boundary.
        loaded_model = _torch.load(_io.BytesIO(weights_payload), map_location="cpu", weights_only=False)
        if not isinstance(loaded_model, dict):
            raise ValueError("Invalid model update payload: expected dict")

        self.active_model = loaded_model
        self.active_version = parsed_version or version or self.active_version
        self._compiled_model = None
        self._scaler_mean = None
        self._scaler_std = None
        self._build_runtime_model()
