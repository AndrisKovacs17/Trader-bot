"""
Offline Training Engine (Adapter Layer)

Responsible for:
- Training KLA predictor on historical data
- Exporting trained model weights
- Pushing updates to the production system via IModelUpdatePort
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import io
import math
import random
from typing import TYPE_CHECKING, Any

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None

from core.ml.services import IModelUpdatePort
from core.ml.feature_engineering import build_trade_feature_rows
from core.ml.kca_mamba import KCAMambaBlock, KCAMambaStack

if TYPE_CHECKING:
    from core.domain.models import Instrument


@dataclass(slots=True)
class TrainedModelArtifact:
    version: str
    weights: dict[str, Any]
    metadata: dict[str, Any]


class TrainingEngine:
    """
    Offline training adapter for model development and refinement.
    
    ⚠️ IMPORTANT: This is an OFFLINE training adapter, NOT the production predictor.
    The production predictor receives trained KLA weights via push_model().
    
    Known ports (imports):
    - IModelUpdatePort: for pushing trained weights to production predictor
    
    Typical workflow:
    1. Load historical dataset
    2. train() - iterate over data, accumulate gradients
    3. export_weights() - serialize model for distribution
    4. push_model() - send to production via ModelUpdatePort
    """
    
    def __init__(self) -> None:
        """Initialize training engine with empty state."""
        self._artifact: TrainedModelArtifact | None = None
        self.training_history: list[dict[str, Any]] = []
        self.trained_version: str = "untrained"

    def train(
        self,
        dataset: list[dict[str, Any]],
        instrument: Instrument | None = None,
        epochs: int = 1,
        batch_size: int = 32,
        hidden_size: int | None = None,
        num_layers: int = 2,
        learning_rate: float = 1e-3,
        use_nll_loss: bool = True,
        lookback: int = 32,
        horizon: int = 3,
        kca_heads: int = 4,
        kca_state_dim: int = 16,
        kca_num_layers: int = 3,
        kca_hidden_dim: int = 64,
        kca_slow_stride: int = 12,
        direction_epsilon: float = 5e-5,
        balance_direction_loss: bool = True,
        direction_pos_weight_min: float = 0.5,
        direction_pos_weight_max: float = 6.0,
        cls_logit_l2: float = 1e-4,
        mu_l2_reg: float = 1e-4,
        estimated_fee_bps: float = 10.0,
        net_target: bool = True,
        seed: int | None = 42,
        early_stopping_patience: int = 3,
        min_val_directional_accuracy: float = 0.52,
        min_baseline_improvement: float = 0.02,
        brier_improvement_margin: float = 0.0,
        collapse_max_class_share: float = 0.90,
        collapse_min_side_share: float = 0.05,
        run_simple_baselines: bool = True,
        baseline_epochs: int = 2,
        cls_loss_weight: float = 2.0,
        nll_loss_clip: float = -10.0,
        head_dropout: float = 0.0,
        weight_decay: float = 0.01,
        label_smoothing: float = 0.0,
        sequence_stride: int = 1,
        find_lr: bool = False,
    ) -> dict[str, Any]:
        """
        Train KLA predictor on historical dataset (OFFLINE).
        
        Args:
            dataset: List of dicts with keys like:
                    - 'price': float
                    - 'return': float
                    - 'volume': float
                    - 'ts': datetime
            instrument: Target instrument (optional)
                       For multi-symbol training, specify instrument explicitly
                       For single-symbol or generic models, None is acceptable
            epochs: Number of training iterations over the dataset
            batch_size: Mini-batch size
            hidden_size: Legacy parameter (unused in KLA-only mode)
            num_layers: Legacy parameter (unused in KLA-only mode)
            learning_rate: Optimizer learning rate
            use_nll_loss: If True, train with uncertainty-aware NLL-style loss
            lookback: Number of past timesteps per sample
            horizon: Future return horizon in steps
            kca_heads: Number of KLA heads
            kca_state_dim: KLA state dimension per head
        
        Returns:
            dict with training metadata:
            - 'epochs': int
            - 'samples': int
            - 'batches': int
            - 'loss': float
            - 'version': str (unique UTC timestamp)
        """
        if not dataset:
            raise ValueError("Dataset cannot be empty")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if lookback < 4:
            raise ValueError(f"lookback must be >= 4, got {lookback}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if kca_heads <= 0:
            raise ValueError(f"kca_heads must be positive, got {kca_heads}")
        if kca_state_dim <= 0:
            raise ValueError(f"kca_state_dim must be positive, got {kca_state_dim}")
        if direction_epsilon < 0:
            raise ValueError(f"direction_epsilon must be >= 0, got {direction_epsilon}")
        if direction_pos_weight_min <= 0:
            raise ValueError(f"direction_pos_weight_min must be > 0, got {direction_pos_weight_min}")
        if direction_pos_weight_max < direction_pos_weight_min:
            raise ValueError(
                f"direction_pos_weight_max must be >= direction_pos_weight_min, got {direction_pos_weight_max} < {direction_pos_weight_min}"
            )
        if cls_logit_l2 < 0:
            raise ValueError(f"cls_logit_l2 must be >= 0, got {cls_logit_l2}")
        if mu_l2_reg < 0:
            raise ValueError(f"mu_l2_reg must be >= 0, got {mu_l2_reg}")
        if estimated_fee_bps < 0:
            raise ValueError(f"estimated_fee_bps must be >= 0, got {estimated_fee_bps}")
        if early_stopping_patience < 0:
            raise ValueError(f"early_stopping_patience must be >= 0, got {early_stopping_patience}")
        if min_baseline_improvement < -1.0:
            raise ValueError(f"min_baseline_improvement must be >= -1.0, got {min_baseline_improvement}")
        if brier_improvement_margin < -1.0:
            raise ValueError(f"brier_improvement_margin must be >= -1.0, got {brier_improvement_margin}")
        if not (0.0 <= min_val_directional_accuracy <= 1.0):
            raise ValueError(
                "min_val_directional_accuracy must be in [0, 1], "
                f"got {min_val_directional_accuracy}"
            )
        if not (0.0 <= collapse_max_class_share <= 1.0):
            raise ValueError(f"collapse_max_class_share must be in [0,1], got {collapse_max_class_share}")
        if not (0.0 <= collapse_min_side_share <= 1.0):
            raise ValueError(f"collapse_min_side_share must be in [0,1], got {collapse_min_side_share}")
        if baseline_epochs <= 0:
            raise ValueError(f"baseline_epochs must be positive, got {baseline_epochs}")
        if not use_nll_loss:
            raise ValueError("Training must use NLL loss. Set use_nll_loss=True.")
        if torch is None or nn is None:
            raise ImportError("TrainingEngine requires PyTorch. Install torch to use training.")

        if seed is not None:
            random.seed(int(seed))
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        total_samples = len(dataset)

        rows: list[dict[str, Any]] = []
        for idx, row in enumerate(dataset):
            price = float(row.get("price", 0.0) or 0.0)
            # Only pass "return" to feature engineering when the caller explicitly
            # provides it.  When absent, leave it out so build_trade_feature_rows
            # computes ret = (close - prev_close) / prev_close from the price
            # series — which is the correct per-bar momentum signal.
            # Passing 0.0 would make the entire ret feature a flat zero vector,
            # killing the most basic momentum signal in the feature set.
            _raw_ret = row.get("return", None)
            volume = float(row.get("volume", row.get("qty", 0.0)) or 0.0)
            ts = row.get("ts_ms", row.get("timestamp", row.get("ts", idx * 1000)))
            if isinstance(ts, datetime):
                ts_ms = int(ts.timestamp() * 1000)
            else:
                ts_ms = int(ts) if ts is not None else idx * 1000
            taker_buy_vol = float(row.get("taker_buy_vol", -1.0) or -1.0)
            funding_rate = float(row.get("funding_rate", 0.0) or 0.0)
            open_p = float(row.get("open", row.get("o", price)) or price)
            high_p = float(row.get("high", row.get("h", price)) or price)
            low_p  = float(row.get("low",  row.get("l", price)) or price)
            row_dict: dict[str, Any] = {
                "price": price,
                "volume": volume,
                "ts_ms": max(ts_ms, 0),
                "taker_buy_vol": taker_buy_vol,
                "funding_rate": funding_rate,
                "o": open_p,
                "h": high_p,
                "l": low_p,
            }
            if _raw_ret is not None:
                row_dict["return"] = float(_raw_ret or 0.0)
            rows.append(row_dict)

        feature_rows, feature_names = build_trade_feature_rows(rows)
        if len(feature_rows) <= lookback + horizon:
            raise ValueError(
                f"Dataset too short for lookback={lookback}, horizon={horizon}. "
                f"Need > {lookback + horizon}, got {len(feature_rows)}"
            )

        x_all = torch.tensor(feature_rows, dtype=torch.float32)
        price_series = torch.tensor([float(item["price"]) for item in rows], dtype=torch.float32)

        fee_rate = max(float(estimated_fee_bps), 0.0) / 10_000.0
        roundtrip_fee_rate = 2.0 * fee_rate

        _stride = max(1, int(sequence_stride))
        n_rows = len(feature_rows)

        # flat_threshold: dead-zone for UP/DOWN/FLAT labelling. With net_target=True
        # the natural dead-zone is the roundtrip fee; with net_target=False use epsilon.
        flat_threshold = max(float(direction_epsilon), roundtrip_fee_rate if net_target else float(direction_epsilon))

        # ── Leak-free train/val split ─────────────────────────────────────────
        # Build windows SEPARATELY per split to prevent validation leakage.
        # Old approach: build all N windows from all rows, slice at 80% → val windows
        # near the boundary share lookback history with train windows (data leak).
        # Fix: insert a gap of exactly `lookback` raw rows at the split boundary so
        # no raw row used in any val window's history appears in any training window.
        raw_train_end = max(lookback + horizon + 1, int(n_rows * 0.8))
        raw_val_start = min(n_rows - lookback - horizon - 1, raw_train_end + lookback)

        def _make_windows(
            x_block: torch.Tensor,
            p_block: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            n_b = x_block.shape[0]
            n_seq_b = n_b - lookback - horizon
            if n_seq_b <= 0:
                z = torch.zeros(0, lookback, x_block.shape[1])
                return z, torch.zeros(0, 1), torch.zeros(0, dtype=torch.long), torch.zeros(0, 1)
            n_win = n_seq_b // _stride
            x_w = x_block.unfold(0, lookback, _stride)[:n_win].permute(0, 2, 1).contiguous()
            idx = torch.arange(0, n_win, dtype=torch.long) * _stride
            p_now = p_block[lookback:lookback + n_seq_b][idx]
            p_fut = p_block[lookback + horizon:lookback + horizon + n_seq_b][idx]
            gross = (p_fut - p_now) / p_now.clamp(min=1e-9)
            tgt = gross.sign() * (gross.abs() - roundtrip_fee_rate).clamp(min=0.0) if net_target else gross
            y_mu_w = tgt.unsqueeze(-1)
            # Binary: UP=1 if net return > 0, DOWN=0 otherwise. No dead-zone FLAT class.
            y_cls_w = (y_mu_w.squeeze(-1) > 0).long()
            return x_w, y_mu_w, y_cls_w, gross.unsqueeze(-1)

        x_train_raw, y_mu_train, y_cls_train, y_gross_train = _make_windows(
            x_all[:raw_train_end], price_series[:raw_train_end]
        )
        x_val_raw, y_mu_val, y_cls_val, _ = _make_windows(
            x_all[raw_val_start:], price_series[raw_val_start:]
        )

        train_size = int(x_train_raw.shape[0])
        val_size = int(x_val_raw.shape[0])
        sequence_samples = train_size + val_size
        if train_size < 64:
            raise ValueError(
                f"Too few training sequence samples ({train_size}). "
                "Increase dataset size or reduce lookback/horizon."
            )

        if hidden_size is None:
            hidden_size = 0

        num_batches = max(1, math.ceil(train_size / batch_size))

        x_train_flat = x_train_raw.reshape(-1, x_train_raw.shape[-1])
        x_mean = x_train_flat.mean(dim=0, keepdim=True)
        x_std = x_train_flat.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        x_train = (x_train_raw - x_mean) / x_std
        x_val = (x_val_raw - x_mean) / x_std

        train_mu_mean = float(y_mu_train.mean().item()) if y_mu_train.numel() > 0 else 0.0
        train_mu_std = float(y_mu_train.std().item()) if y_mu_train.numel() > 1 else 0.0
        train_abs_move_median = float(torch.median(torch.abs(y_gross_train)).item()) if y_gross_train.numel() > 0 else 0.0
        horizon_fee_coverage = train_abs_move_median / roundtrip_fee_rate if roundtrip_fee_rate > 0 else float("inf")

        feature_dim = int(x_train.shape[2])
        class_count = 2  # Binary: DOWN=0, UP=1  (FLAT removed)
        _head_hidden = 64
        kca_stack = KCAMambaStack(
            feature_dim=feature_dim,
            hidden_dim=kca_hidden_dim,
            num_layers=kca_num_layers,
            heads=kca_heads,
            d_state=kca_state_dim,
            slow_stride=kca_slow_stride,
        )
        mu_head = nn.Sequential(nn.Linear(kca_hidden_dim, _head_hidden), nn.SiLU(), nn.Linear(_head_hidden, 1))
        up_head = nn.Sequential(nn.Linear(kca_hidden_dim, _head_hidden), nn.SiLU(), nn.Linear(_head_hidden, class_count))
        var_head = nn.Sequential(nn.Linear(kca_hidden_dim, _head_hidden), nn.SiLU(), nn.Linear(_head_hidden, 1))
        model = nn.ModuleDict({"kca_stack": kca_stack, "mu_head": mu_head, "up_head": up_head, "var_head": var_head})

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        # Keep full tensors on CPU to avoid large one-shot GPU allocations.
        # Only move minibatches to GPU during train/eval steps.
        non_blocking_transfer = device == "cuda"
        eval_batch_size = max(1, int(batch_size))

        # ── Fastai-style LR finder (optional) ────────────────────────────────
        # Same algorithm as _find_lr_fastai in ltsf_benchmark.py, but uses the
        # actual NLL+CE forward pass of this model (nn.ModuleDict has no .forward).
        if find_lr and train_size >= batch_size:
            import copy as _copy
            _lr_start, _lr_end, _num_it = 1e-7, 1.0, 100
            _beta_ema = 0.98
            _model_clone = _copy.deepcopy(model).to(device)
            _opt_clone = torch.optim.Adam(_model_clone.parameters(), lr=_lr_start)
            _avg_loss_ema = 0.0
            _best_smooth = float("inf")
            _lrs: list[float] = []
            _losses: list[float] = []
            _perm_lr = torch.randperm(train_size)
            _ptr = 0
            print("[LR-find] Probing 1e-7 → 1.0 …", flush=True)
            for _i in range(_num_it):
                _lr_i = _lr_start * (_lr_end / _lr_start) ** (_i / _num_it)
                for _pg in _opt_clone.param_groups:
                    _pg["lr"] = _lr_i
                if _ptr + batch_size > train_size:
                    _perm_lr = torch.randperm(train_size)
                    _ptr = 0
                _idx = _perm_lr[_ptr: _ptr + batch_size]
                _ptr += batch_size
                _xb = x_train[_idx].to(device)
                _yb_mu = y_mu_train[_idx].to(device)
                _yb_cls = y_cls_train[_idx].to(device)
                _opt_clone.zero_grad(set_to_none=True)
                _y_seq, _, _, _ = _model_clone["kca_stack"](_xb)
                _h = _y_seq[:, -1, :]
                _pred_mu = _model_clone["mu_head"](_h)
                _pred_var = torch.nn.functional.softplus(_model_clone["var_head"](_h)).clamp(1e-6, 10.0)
                _pred_logits = _model_clone["up_head"](_h)
                _nll = ((_pred_mu - _yb_mu).pow(2) / (2.0 * _pred_var) + 0.5 * torch.log(_pred_var)).mean()
                if nll_loss_clip > -1e6:
                    _nll = _nll.clamp(min=float(nll_loss_clip))
                _ce_probe = nn.CrossEntropyLoss()(_pred_logits, _yb_cls)
                _loss = _nll + float(cls_loss_weight) * _ce_probe
                if not torch.isfinite(_loss):
                    break
                _loss.backward()
                torch.nn.utils.clip_grad_norm_(_model_clone.parameters(), 1.0)
                _opt_clone.step()
                _avg_loss_ema = _beta_ema * _avg_loss_ema + (1.0 - _beta_ema) * _loss.item()
                _smooth = _avg_loss_ema / (1.0 - _beta_ema ** (_i + 1))
                if _smooth < _best_smooth:
                    _best_smooth = _smooth
                elif _smooth > 4.0 * _best_smooth:
                    break
                _lrs.append(_lr_i)
                _losses.append(_smooth)
            del _model_clone, _opt_clone
            # Valley suggestion: trim edges, find longest decreasing run, take ~2/3 through
            _trim_s = max(0, _num_it // 10)
            _trim_e = max(_trim_s + 1, len(_lrs) - 5)
            _lt = _lrs[_trim_s:_trim_e]
            _ls = _losses[_trim_s:_trim_e]
            if len(_ls) >= 3:
                _dp = [1] * len(_ls)
                _prev = [-1] * len(_ls)
                for _j in range(1, len(_ls)):
                    for _k in range(_j):
                        if _ls[_k] > _ls[_j] and _dp[_k] + 1 > _dp[_j]:
                            _dp[_j] = _dp[_k] + 1
                            _prev[_j] = _k
                _best_end = max(range(len(_dp)), key=lambda x: _dp[x])
                _cur = _best_end
                _run_indices: list[int] = []
                while _cur >= 0:
                    _run_indices.append(_cur)
                    _cur = _prev[_cur]
                _run_indices.reverse()
                _valley_idx = _run_indices[int(len(_run_indices) * 2 // 3)]
                _suggested_lr = _lt[_valley_idx]
            else:
                _suggested_lr = float(learning_rate)
            _suggested_lr = float(max(1e-6, min(_suggested_lr, 1e-2)))
            print(f"[LR-find] Suggested LR: {_suggested_lr:.2e}  (was {float(learning_rate):.2e})", flush=True)
            learning_rate = _suggested_lr

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
        # Warmup prevents Beta-NLL exploding gradients on epoch 1 when sigma is random.
        _warmup_epochs = min(3, max(1, epochs // 5))
        def _lr_lambda(epoch_idx: int) -> float:
            if epoch_idx < _warmup_epochs:
                return float(epoch_idx + 1) / float(_warmup_epochs)
            cos_progress = (epoch_idx - _warmup_epochs) / max(1, epochs - _warmup_epochs)
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * cos_progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

        class_counts = torch.bincount(y_cls_train, minlength=class_count).float()
        class_total = float(class_counts.sum().item())
        if balance_direction_loss and class_total > 0:
            raw_weights = class_total / (float(class_count) * class_counts.clamp_min(1.0))
            class_weights = raw_weights.clamp(float(direction_pos_weight_min), float(direction_pos_weight_max))
        else:
            class_weights = torch.ones(class_count, dtype=torch.float32)
        ce = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=float(label_smoothing))

        train_class_priors = (class_counts / max(class_total, 1.0)).tolist()

        def _class_to_up_score(labels: torch.Tensor) -> torch.Tensor:
            # Binary: UP=1 → score 1.0, DOWN=0 → score 0.0
            return labels.float()

        baseline_directional_accuracy = float("nan")
        baseline_brier_up = float("nan")
        baseline_mae_return = float("nan")
        if x_val_raw.shape[0] > 0:
            baseline_mu = x_val_raw[:, -1, 0:1]
            # Binary baseline: UP if last return > 0, else DOWN
            baseline_cls = (baseline_mu.squeeze(-1) > 0).long().to(y_cls_val.device)
            baseline_directional_accuracy = float((baseline_cls == y_cls_val).float().mean().item())
            _bl_mu_vals = baseline_mu.squeeze(-1)
            _bl_cls_dist = torch.bincount(baseline_cls, minlength=2).float() / max(float(baseline_cls.shape[0]), 1)
            _tgt_cls_dist = torch.bincount(y_cls_val, minlength=2).float() / max(float(y_cls_val.shape[0]), 1)
            print(
                f"  [baseline]  last-ret predictor: acc={baseline_directional_accuracy:.4f}"
                f"  ret_range=[{float(_bl_mu_vals.min()):.4e}, {float(_bl_mu_vals.max()):.4e}]"
                f"  →pred DOWN={float(_bl_cls_dist[0]):.2f} UP={float(_bl_cls_dist[1]):.2f}"
                f"  |  true DOWN={float(_tgt_cls_dist[0]):.2f} UP={float(_tgt_cls_dist[1]):.2f}",
                flush=True,
            )
            baseline_prob_up = _class_to_up_score(baseline_cls)
            true_up_score = _class_to_up_score(y_cls_val)
            baseline_brier_up = float(((baseline_prob_up - true_up_score) ** 2).mean().item())
            baseline_mae_return = float(torch.abs(baseline_mu.to(y_mu_val.device) - y_mu_val).mean().item())

        baseline_models: dict[str, dict[str, Any]] = {}
        baseline_reference_accuracy = baseline_directional_accuracy if not math.isnan(baseline_directional_accuracy) else 0.0
        baseline_reference_brier = baseline_brier_up if not math.isnan(baseline_brier_up) else float("inf")

        def _finalize_classifier_metrics(
            total_samples: int,
            correct_samples: int,
            brier_sum: float,
            pred_counts: torch.Tensor,
        ) -> dict[str, Any]:
            if total_samples <= 0:
                return {
                    "val_directional_accuracy": float("nan"),
                    "val_brier_up": float("nan"),
                    "pred_class_share": [float("nan")] * class_count,
                    "collapse_detected": True,
                    "collapse_reason": "no-validation-samples",
                }

            val_acc = float(correct_samples) / float(total_samples)
            val_brier_up = float(brier_sum) / float(total_samples)

            shares = (pred_counts / max(float(pred_counts.sum().item()), 1.0)).tolist()
            max_class_share = max(shares) if shares else 1.0
            # Binary: shares[0]=DOWN, shares[1]=UP
            up_share = float(shares[1]) if len(shares) > 1 else 0.0
            down_share = float(shares[0]) if len(shares) > 0 else 0.0
            collapse_reasons: list[str] = []
            if max_class_share > float(collapse_max_class_share):
                collapse_reasons.append("max-class-share")
            if min(up_share, down_share) < float(collapse_min_side_share):
                collapse_reasons.append("one-sided")

            return {
                "val_directional_accuracy": val_acc,
                "val_brier_up": val_brier_up,
                "pred_class_share": [float(x) for x in shares],
                "collapse_detected": len(collapse_reasons) > 0,
                "collapse_reason": ",".join(collapse_reasons) if collapse_reasons else "",
            }

        if run_simple_baselines and x_val.shape[0] > 0:
            baseline_train_size = min(int(x_train.shape[0]), 20000)
            baseline_x_train = x_train[:baseline_train_size]
            baseline_y_train = y_cls_train[:baseline_train_size]

            class _LogisticBaseline(nn.Module):
                def __init__(self, lookback_size: int, feat_dim: int, classes: int) -> None:
                    super().__init__()
                    self.fc = nn.Linear(lookback_size * feat_dim, classes)

                def forward(self, xb: torch.Tensor) -> torch.Tensor:
                    return self.fc(xb.reshape(xb.shape[0], -1))

            class _MLPBaseline(nn.Module):
                def __init__(self, lookback_size: int, feat_dim: int, classes: int) -> None:
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(lookback_size * feat_dim, 64),
                        nn.SiLU(),
                        nn.Linear(64, classes),
                    )

                def forward(self, xb: torch.Tensor) -> torch.Tensor:
                    return self.net(xb.reshape(xb.shape[0], -1))

            class _GRUBaseline(nn.Module):
                def __init__(self, feat_dim: int, classes: int) -> None:
                    super().__init__()
                    self.gru = nn.GRU(input_size=feat_dim, hidden_size=32, batch_first=True)
                    self.out = nn.Linear(32, classes)

                def forward(self, xb: torch.Tensor) -> torch.Tensor:
                    yb, _ = self.gru(xb)
                    return self.out(yb[:, -1, :])

            class _LSTMBaseline(nn.Module):
                def __init__(self, feat_dim: int, classes: int) -> None:
                    super().__init__()
                    self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=32, batch_first=True)
                    self.out = nn.Linear(32, classes)

                def forward(self, xb: torch.Tensor) -> torch.Tensor:
                    _, (h, _) = self.lstm(xb)
                    return self.out(h[-1])

            class _PlainMambaBaseline(nn.Module):
                """Simplified SSM (no KLA gating): learned per-channel exponential decay."""
                def __init__(self, feat_dim: int, hidden_dim: int, classes: int) -> None:
                    super().__init__()
                    self.in_proj = nn.Linear(feat_dim, hidden_dim)
                    # Learnable per-channel decay rates (initialised to ~0.9 decay)
                    self.A_log = nn.Parameter(torch.full((hidden_dim,), -2.0))
                    self.out_norm = nn.LayerNorm(hidden_dim)
                    self.head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.SiLU(), nn.Linear(64, classes))

                def forward(self, xb: torch.Tensor) -> torch.Tensor:
                    x = self.in_proj(xb)  # [B, T, H]
                    A = torch.sigmoid(self.A_log)  # decay in (0,1), per channel
                    # Sequential scan over time (correct SSM recurrence: h_t = A*h_{t-1} + (1-A)*x_t)
                    h = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
                    for t in range(x.shape[1]):
                        h = A * h + (1.0 - A) * x[:, t, :]
                    return self.head(self.out_norm(h))

            baseline_defs = {
                "logistic": _LogisticBaseline(int(lookback), int(feature_dim), int(class_count)),
                "mlp": _MLPBaseline(int(lookback), int(feature_dim), int(class_count)),
                "gru": _GRUBaseline(int(feature_dim), int(class_count)),
                "lstm": _LSTMBaseline(int(feature_dim), int(class_count)),
                "plain_mamba": _PlainMambaBaseline(int(feature_dim), 48, int(class_count)),
            }

            for baseline_name, baseline_model in baseline_defs.items():
                baseline_model = baseline_model.to(device)
                baseline_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=5e-4)
                baseline_ce = nn.CrossEntropyLoss(weight=class_weights.to(device))

                baseline_model.train()
                for _ in range(int(baseline_epochs)):
                    perm = torch.randperm(int(baseline_x_train.shape[0]))
                    for batch_start in range(0, int(baseline_x_train.shape[0]), batch_size):
                        idx = perm[batch_start: batch_start + batch_size]
                        xb = baseline_x_train[idx].to(device, non_blocking=non_blocking_transfer)
                        yb = baseline_y_train[idx].to(device, non_blocking=non_blocking_transfer)
                        logits = baseline_model(xb)
                        loss = baseline_ce(logits, yb)
                        baseline_optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        baseline_optimizer.step()

                baseline_model.eval()
                with torch.no_grad():
                    total_eval = 0
                    correct_eval = 0
                    brier_sum_eval = 0.0
                    pred_counts_eval = torch.zeros(class_count, dtype=torch.float32)
                    for batch_start in range(0, int(x_val.shape[0]), eval_batch_size):
                        xb = x_val[batch_start: batch_start + eval_batch_size].to(
                            device,
                            non_blocking=non_blocking_transfer,
                        )
                        yb_cls = y_cls_val[batch_start: batch_start + eval_batch_size].to(
                            device,
                            non_blocking=non_blocking_transfer,
                        )
                        val_logits = baseline_model(xb)
                        val_probs = torch.softmax(val_logits, dim=-1)
                        pred_cls = torch.argmax(val_probs, dim=-1)

                        correct_eval += int((pred_cls == yb_cls).sum().item())

                        prob_up = val_probs[:, 1]  # Binary: class 1 = UP
                        true_up_score = _class_to_up_score(yb_cls)
                        brier_sum_eval += float(((prob_up - true_up_score) ** 2).sum().item())

                        pred_counts_eval += torch.bincount(
                            pred_cls.detach().cpu(), minlength=class_count
                        ).float()
                        total_eval += int(yb_cls.shape[0])

                    metrics = _finalize_classifier_metrics(
                        total_samples=total_eval,
                        correct_samples=correct_eval,
                        brier_sum=brier_sum_eval,
                        pred_counts=pred_counts_eval,
                    )

                baseline_models[baseline_name] = metrics
                if not math.isnan(float(metrics["val_directional_accuracy"])):
                    baseline_reference_accuracy = max(
                        baseline_reference_accuracy,
                        float(metrics["val_directional_accuracy"]),
                    )
                if not math.isnan(float(metrics["val_brier_up"])):
                    baseline_reference_brier = min(
                        baseline_reference_brier,
                        float(metrics["val_brier_up"]),
                    )

            # AR (AutoRegressive / ARIMA-like) baseline: fits AR(p) on return feature via lstsq
            # Uses only the first feature column (scaled returns) → linear combination → classify
            # NOTE: AR is informational only — not included in baseline_reference_accuracy
            # (degenerate AR solutions tend to predict FLAT > 80% and dominate the reference)
            try:
                ar_x = baseline_x_train[:, :, 0].cpu().float()  # [N, lookback] — ret feature
                ar_bias = torch.ones(ar_x.shape[0], 1)
                ar_X = torch.cat([ar_x, ar_bias], dim=1)  # [N, lookback+1]
                # Regression targets: DOWN→-1, UP→+1 (binary, no FLAT)
                _cls_to_reg = torch.tensor([-1.0, 1.0])
                ar_y = _cls_to_reg[baseline_y_train.cpu().long()]  # [N]
                ar_result = torch.linalg.lstsq(ar_X, ar_y.unsqueeze(1))
                ar_coef = ar_result.solution  # [lookback+1, 1]

                # Evaluate on val set
                ar_xv = x_val[:, :, 0].cpu().float()
                ar_Xv = torch.cat([ar_xv, torch.ones(ar_xv.shape[0], 1)], dim=1)
                ar_pred_raw = (ar_Xv @ ar_coef).squeeze(1)  # [N_val]
                # Binary threshold at median (balanced split)
                ar_pred_cls = (ar_pred_raw > 0.0).long()
                # Soft probabilities: sigmoid of raw score
                ar_prob_up = torch.sigmoid(ar_pred_raw * 5.0).clamp(0.0, 1.0)

                ar_true_cls = y_cls_val.cpu()
                ar_correct = int((ar_pred_cls == ar_true_cls).sum().item())
                ar_true_up = _class_to_up_score(ar_true_cls)
                ar_brier = float(((ar_prob_up - ar_true_up) ** 2).sum().item())
                ar_pred_counts = torch.bincount(ar_pred_cls.long(), minlength=class_count).float()

                ar_metrics = _finalize_classifier_metrics(
                    total_samples=int(ar_true_cls.shape[0]),
                    correct_samples=ar_correct,
                    brier_sum=ar_brier,
                    pred_counts=ar_pred_counts,
                )
                baseline_models["ar"] = ar_metrics
                # Intentionally NOT updating baseline_reference_accuracy / brier here
                # AR is a reference model only, not a deploy gate criterion
            except Exception:
                pass  # AR fitting can fail on degenerate data; skip silently

        # ── baseline summary ────────────────────────────────────────────────
        print("\n  Baseline models (reference for deploy gate):")
        print(f"    {'Model':<18}  {'ValAcc':>7}  {'Brier':>7}")
        print(f"    {'-'*18}  {'-'*7}  {'-'*7}")
        if not math.isnan(baseline_directional_accuracy):
            print(f"    {'naive_last_ret':<18}  {baseline_directional_accuracy:>7.4f}  {baseline_brier_up:>7.4f}")
        for _bname, _bm in baseline_models.items():
            _bacc = float(_bm.get("val_directional_accuracy", float("nan")))
            _bbrier = float(_bm.get("val_brier_up", float("nan")))
            _ref_marker = " <-- ref" if abs(_bacc - baseline_reference_accuracy) < 1e-9 else ""
            print(f"    {_bname:<18}  {_bacc:>7.4f}  {_bbrier:>7.4f}{_ref_marker}")
        print(f"    => KLA must beat acc={baseline_reference_accuracy:.4f} "
              f"by +{float(min_baseline_improvement):.4f}  (gate: >={float(min_val_directional_accuracy):.4f})")

        def evaluate_model() -> dict[str, Any]:
            with torch.no_grad():
                if x_val.shape[0] <= 0:
                    return {
                        "val_loss": float("nan"),
                        "val_directional_accuracy": float("nan"),
                        "val_brier_up": float("nan"),
                        "val_samples": 0,
                        "pred_class_share": [float("nan")] * class_count,
                        "collapse_detected": True,
                        "collapse_reason": "no-validation-samples",
                    }

                total_eval = 0
                correct_eval = 0
                brier_sum_eval = 0.0
                loss_sum_eval = 0.0
                pred_counts_eval = torch.zeros(class_count, dtype=torch.float32)

                for batch_start in range(0, int(x_val.shape[0]), eval_batch_size):
                    xb = x_val[batch_start: batch_start + eval_batch_size].to(
                        device,
                        non_blocking=non_blocking_transfer,
                    )
                    yb_mu = y_mu_val[batch_start: batch_start + eval_batch_size].to(
                        device,
                        non_blocking=non_blocking_transfer,
                    )
                    yb_cls = y_cls_val[batch_start: batch_start + eval_batch_size].to(
                        device,
                        non_blocking=non_blocking_transfer,
                    )

                    y_val, _, _, _ = model["kca_stack"](xb)
                    h_val = y_val[:, -1, :]
                    pred_mu_val = model["mu_head"](h_val)
                    pred_logits_val = model["up_head"](h_val)
                    pred_probs_val = torch.softmax(pred_logits_val, dim=-1)
                    pred_var_val = torch.nn.functional.softplus(model["var_head"](h_val)).clamp(1e-6, 10.0)

                    mse_part_val = (pred_mu_val - yb_mu).pow(2) / (2.0 * pred_var_val)
                    log_part_val = 0.5 * torch.log(pred_var_val)
                    _beta_val = 0.0  # must match train beta
                    _var_weight_val = pred_var_val.detach().pow(_beta_val)
                    beta_mse_val = _var_weight_val * (pred_mu_val - yb_mu).pow(2) / (2.0 * pred_var_val)
                    nll_val = (beta_mse_val + log_part_val).mean()
                    if nll_loss_clip > -1e6:
                        nll_val = nll_val.clamp(min=float(nll_loss_clip))
                    val_loss = nll_val + float(cls_loss_weight) * ce(pred_logits_val, yb_cls)

                    batch_n = int(yb_cls.shape[0])
                    loss_sum_eval += float(val_loss.detach().cpu().item()) * batch_n

                    pred_cls_val = torch.argmax(pred_probs_val, dim=-1)
                    correct_eval += int((pred_cls_val == yb_cls).sum().item())

                    prob_up_val = pred_probs_val[:, 1]  # Binary: class 1 = UP
                    true_up_score = _class_to_up_score(yb_cls)
                    brier_sum_eval += float(((prob_up_val - true_up_score) ** 2).sum().item())

                    pred_counts_eval += torch.bincount(
                        pred_cls_val.detach().cpu(), minlength=class_count
                    ).float()
                    total_eval += batch_n

                metrics = _finalize_classifier_metrics(
                    total_samples=total_eval,
                    correct_samples=correct_eval,
                    brier_sum=brier_sum_eval,
                    pred_counts=pred_counts_eval,
                )
                return {
                    "val_loss": float(loss_sum_eval / max(total_eval, 1)),
                    "val_directional_accuracy": float(metrics["val_directional_accuracy"]),
                    "val_brier_up": float(metrics["val_brier_up"]),
                    "val_samples": int(total_eval),
                    "pred_class_share": metrics["pred_class_share"],
                    "collapse_detected": bool(metrics["collapse_detected"]),
                    "collapse_reason": str(metrics["collapse_reason"]),
                }

        best_score = -float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        best_train_loss = float("nan")

        best_gate_score = -float("inf")
        best_gate_state: dict[str, torch.Tensor] | None = None
        best_gate_epoch = 0

        epochs_without_improve = 0
        epoch_history: list[dict[str, Any]] = []
        epochs_trained = 0

        # ── training header ───────────────────────────────────────────────────
        print(
            f"\n{'='*72}\n"
            f"  KCA-Mamba Training  |  epochs={epochs}  batches/epoch={num_batches}"
            f"  |  train={train_size}  val={val_size}  feat={feature_dim}\n"
            f"  device={device}  lookback={lookback}  horizon={horizon}"
            f"  |  baseline_acc={baseline_reference_accuracy:.4f}  baseline_brier={baseline_reference_brier:.4f}\n"
            f"{'='*72}"
        )
        print(f"  {'Ep':>4}  {'TrainLoss':>10}  {'ValLoss':>9}  {'ValAcc':>7}  "
              f"{'AccGain':>8}  {'Brier':>7}  {'Gate':>5}  {'ES':>4}  {'Note'}")
        print(f"  {'-'*4}  {'-'*10}  {'-'*9}  {'-'*7}  "
              f"{'-'*8}  {'-'*7}  {'-'*5}  {'-'*4}  {'-'*20}")

        for epoch in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(train_size)
            epoch_loss_sum = 0.0
            epoch_batches = 0
            for batch_start in range(0, train_size, batch_size):
                idx = perm[batch_start: batch_start + batch_size]
                xb = x_train[idx].to(device, non_blocking=non_blocking_transfer)
                yb_mu = y_mu_train[idx].to(device, non_blocking=non_blocking_transfer)
                yb_cls = y_cls_train[idx].to(device, non_blocking=non_blocking_transfer)

                y_seq, _, _, _ = model["kca_stack"](xb)
                h = y_seq[:, -1, :]
                if head_dropout > 0.0:
                    h = torch.nn.functional.dropout(h, p=float(head_dropout), training=True)
                pred_mu = model["mu_head"](h)
                pred_logits = model["up_head"](h)
                pred_var = torch.nn.functional.softplus(model["var_head"](h)).clamp(1e-6, 10.0)

                # (A) Standard NLL (beta=0): for financial data where residuals ~0.001
                # Beta-NLL beta=0.5 was designed for models where mu≈y (image regression),
                # but drives var_optimal = residual^4 = (0.001)^4 = 1e-12 for financial returns
                # → clamp → sigma permanently dead. Standard NLL (beta=0) gives var = residual^2
                # = σ_data^2 ≈ 1e-6…1e-4 which actually tracks volatility regimes.
                _beta = 0.0
                _var_weight = pred_var.detach().pow(_beta)  # =1 (no reweighting, pure NLL)
                beta_mse = _var_weight * (pred_mu - yb_mu).pow(2) / (2.0 * pred_var)
                log_part = 0.5 * torch.log(pred_var)
                nll_loss = (beta_mse + log_part).mean()
                if nll_loss_clip > -1e6:
                    nll_loss = nll_loss.clamp(min=float(nll_loss_clip))
                if mu_l2_reg > 0:
                    nll_loss = nll_loss + float(mu_l2_reg) * pred_mu.pow(2).mean()

                # (B) Classification
                cls_loss = float(cls_loss_weight) * ce(pred_logits, yb_cls)
                if cls_logit_l2 > 0:
                    cls_loss = cls_loss + float(cls_logit_l2) * pred_logits.pow(2).mean()

                loss = nll_loss + cls_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss_sum += float(loss.detach().cpu().item())
                epoch_batches += 1

            epochs_trained = epoch
            train_epoch_loss = epoch_loss_sum / max(epoch_batches, 1)

            model.eval()
            val_metrics = evaluate_model()
            val_acc = float(val_metrics["val_directional_accuracy"])
            val_brier = float(val_metrics["val_brier_up"])
            val_loss_value = float(val_metrics["val_loss"])
            val_samples = int(val_metrics["val_samples"])
            collapse_detected = bool(val_metrics["collapse_detected"])
            collapse_reason = str(val_metrics["collapse_reason"])

            acc_gain = val_acc - baseline_reference_accuracy if not math.isnan(val_acc) and not math.isnan(baseline_reference_accuracy) else -float("inf")
            brier_gain = baseline_reference_brier - val_brier if not math.isnan(val_brier) and not math.isnan(baseline_reference_brier) else -float("inf")

            meets_gate = (
                val_samples > 0
                and not math.isnan(val_acc)
                and not math.isnan(val_brier)
                and val_acc >= float(min_val_directional_accuracy)
                and acc_gain >= float(min_baseline_improvement)
                and brier_gain >= float(brier_improvement_margin)
                and (not collapse_detected)
            )

            deploy_score = (
                (0.0 if math.isinf(acc_gain) else acc_gain)
                + (0.0 if math.isinf(brier_gain) else brier_gain)
                - (1.0 if collapse_detected else 0.0)
            )

            if deploy_score > best_score:
                best_score = deploy_score
                best_epoch = epoch
                best_train_loss = train_epoch_loss
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if meets_gate and deploy_score > best_gate_score:
                best_gate_score = deploy_score
                best_gate_epoch = epoch
                best_gate_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

            epoch_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_epoch_loss,
                    "val_loss": val_loss_value,
                    "val_directional_accuracy": val_acc,
                    "val_brier_up": val_brier,
                    "val_acc_gain_vs_baseline": acc_gain,
                    "val_brier_gain_vs_baseline": brier_gain,
                    "collapse_detected": collapse_detected,
                    "collapse_reason": collapse_reason,
                    "meets_deploy_gate": meets_gate,
                    "deploy_score": deploy_score,
                }
            )

            # ── per-epoch console output ───────────────────────────────────────
            _gate_str = "  OK " if meets_gate else "  -- "
            _es_str = f"{epochs_without_improve}/{early_stopping_patience}" if early_stopping_patience > 0 else "off"
            _note_parts = []
            if epoch == best_epoch:
                _note_parts.append("best")
            if epoch == best_gate_epoch:
                _note_parts.append("gate*")
            if collapse_detected:
                _note_parts.append(f"COLLAPSE:{collapse_reason}")
            _note = ", ".join(_note_parts) if _note_parts else ""
            print(
                f"  {epoch:>4}  {train_epoch_loss:>10.5f}  {val_loss_value:>9.5f}  "
                f"{val_acc:>7.4f}  {acc_gain:>+8.4f}  {val_brier:>7.4f}  "
                f"{_gate_str:>5}  {_es_str:>4}  {_note}",
                flush=True,
            )

            scheduler.step()
            if early_stopping_patience > 0 and epochs_without_improve >= early_stopping_patience:
                print(f"\n  [EarlyStopping] Stopped at epoch {epoch} "
                      f"(no improvement for {early_stopping_patience} epochs)")
                break

        print(f"{'='*72}")

        if best_state is None:
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            best_epoch = epochs_trained

        deployable = best_gate_state is not None
        if deployable and best_gate_state is not None:
            selected_state = best_gate_state
            selected_epoch = best_gate_epoch
        else:
            selected_state = best_state
            selected_epoch = best_epoch

        # Validate checkpoint key/shape compatibility before loading.
        # strict=False silently ignores missing/unexpected keys → partial random weights.
        _ckpt_keys = set(selected_state.keys())
        _model_keys = set(model.state_dict().keys())
        _missing_in_ckpt = sorted(_model_keys - _ckpt_keys)
        _unexpected_in_ckpt = sorted(_ckpt_keys - _model_keys)
        _shape_mismatches = [
            (k, tuple(model.state_dict()[k].shape), tuple(selected_state[k].shape))
            for k in _ckpt_keys & _model_keys
            if model.state_dict()[k].shape != selected_state[k].shape
        ]
        if _missing_in_ckpt or _unexpected_in_ckpt or _shape_mismatches:
            raise RuntimeError(
                f"Checkpoint incompatible with model — "
                f"missing={_missing_in_ckpt}, unexpected={_unexpected_in_ckpt}, "
                f"shape_mismatches={_shape_mismatches}"
            )
        model.load_state_dict(selected_state, strict=True)
        model.eval()
        final_val = evaluate_model()

        state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}

        train_positive_ratio = float(train_class_priors[2]) if len(train_class_priors) >= 3 else 0.5

        weights = {
            "predictor": {
                "state_dict": state_dict,
                "hyperparams": {
                    "model_type": "kca",
                    "input_size": feature_dim,
                    "feature_names": feature_names,
                    "class_count": class_count,
                    "kca_heads": int(kca_heads),
                    "kca_state_dim": int(kca_state_dim),
                    "kca_num_layers": int(kca_num_layers),
                    "kca_hidden_dim": int(kca_hidden_dim),
                    "kca_slow_stride": int(kca_slow_stride),
                    "lookback": int(lookback),
                    "horizon": int(horizon),
                    "device": device,
                    "loss": "nll" if use_nll_loss else "mse",
                    "target_mode": "net_of_fee" if net_target else "raw_return",
                    "estimated_fee_bps": float(estimated_fee_bps),
                    "roundtrip_fee_rate": float(roundtrip_fee_rate),
                    "flat_threshold": float(flat_threshold),
                    "train_positive_ratio": train_positive_ratio,
                    "train_class_priors": [float(x) for x in train_class_priors],
                    "train_mu_mean": train_mu_mean,
                    "train_mu_std": train_mu_std,
                    "direction_epsilon": float(direction_epsilon),
                    "effective_direction_epsilon": float(flat_threshold),
                    "direction_filter_mode": "three_class_fee_epsilon",
                    "direction_target_source": "net_return" if net_target else "gross_return",
                    "seed": int(seed) if seed is not None else None,
                },
                "scaler": {
                    "mean": x_mean.detach().cpu().view(-1).tolist(),
                    "std": x_std.detach().cpu().view(-1).tolist(),
                },
            },
        }

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.trained_version = f"offline-{timestamp}-e{selected_epoch}"

        final_acc = float(final_val["val_directional_accuracy"])
        final_brier = float(final_val["val_brier_up"])
        final_collapse = bool(final_val["collapse_detected"])
        final_collapse_reason = str(final_val["collapse_reason"])
        acc_gain_final = (
            final_acc - baseline_reference_accuracy
            if not math.isnan(final_acc) and not math.isnan(baseline_reference_accuracy)
            else float("nan")
        )
        brier_gain_final = (
            baseline_reference_brier - final_brier
            if not math.isnan(final_brier) and not math.isnan(baseline_reference_brier)
            else float("nan")
        )

        if deployable:
            deploy_reason = "meets-baseline-brier-collapse-gates"
        else:
            fail_reasons: list[str] = []
            if math.isnan(final_acc) or final_acc < float(min_val_directional_accuracy):
                fail_reasons.append("val-acc-floor")
            if math.isnan(acc_gain_final) or acc_gain_final < float(min_baseline_improvement):
                fail_reasons.append("baseline-acc-gap")
            if math.isnan(brier_gain_final) or brier_gain_final < float(brier_improvement_margin):
                fail_reasons.append("brier-gap")
            if final_collapse:
                fail_reasons.append(f"collapse:{final_collapse_reason or 'yes'}")
            deploy_reason = ",".join(fail_reasons) if fail_reasons else "no-gate-pass"

        # ── final summary ─────────────────────────────────────────────────────
        _deploy_label = "DEPLOYABLE  " if deployable else "NOT deployable"
        _class_shares = final_val.get("pred_class_share", [])
        _shares_str = (
            f"DOWN={_class_shares[0]:.3f}  UP={_class_shares[1]:.3f}"
            if len(_class_shares) >= 2 else "n/a"
        )
        print(
            f"\n  Result : [{_deploy_label}]  epoch={selected_epoch}/{epochs_trained}"
            f"  version={self.trained_version}"
            f"\n  Final  : acc={final_acc:.4f} ({acc_gain_final:+.4f} vs baseline)"
            f"  brier={final_brier:.4f} ({brier_gain_final:+.4f} vs baseline)"
            f"\n  Preds  : {_shares_str}"
            + (f"\n  GATE FAIL: {deploy_reason}" if not deployable else "")
            + (f"\n  COLLAPSE: {final_collapse_reason}" if final_collapse else "")
        )
        print(f"{'='*72}\n")

        result: dict[str, Any] = {
            "epochs": epochs_trained,
            "epochs_requested": epochs,
            "best_epoch": selected_epoch,
            "samples": sequence_samples * epochs_trained,
            "raw_samples": total_samples,
            "train_samples": train_size,
            "val_samples": int(final_val["val_samples"]),
            "batches": num_batches,
            "loss_value": best_train_loss,
            "val_loss_value": float(final_val["val_loss"]),
            "val_directional_accuracy": final_acc,
            "val_brier_up": final_brier,
            "val_pred_class_share": final_val["pred_class_share"],
            "val_collapse_detected": final_collapse,
            "val_collapse_reason": final_collapse_reason,
            "val_gate_min_directional_accuracy": float(min_val_directional_accuracy),
            "val_gate_min_baseline_improvement": float(min_baseline_improvement),
            "val_gate_brier_improvement_margin": float(brier_improvement_margin),
            "deployable": bool(deployable),
            "deploy_reason": deploy_reason,
            "version": self.trained_version,
            "device": device,
            "kca_heads": kca_heads,
            "kca_state_dim": kca_state_dim,
            "lookback": lookback,
            "horizon": horizon,
            "direction_epsilon": direction_epsilon,
            "effective_direction_epsilon": float(flat_threshold),
            "direction_filter_mode": "three_class_fee_epsilon",
            "direction_target_source": "net_return" if net_target else "gross_return",
            "target_mode": "net_of_fee" if net_target else "raw_return",
            "estimated_fee_bps": float(estimated_fee_bps),
            "roundtrip_fee_rate": float(roundtrip_fee_rate),
            "horizon_fee_coverage": horizon_fee_coverage,
            "train_abs_move_median": train_abs_move_median,
            "balance_direction_loss": balance_direction_loss,
            "direction_class_weights": class_weights.detach().cpu().tolist(),
            "train_positive_ratio": train_positive_ratio,
            "train_class_priors": [float(x) for x in train_class_priors],
            "baseline_directional_accuracy": baseline_directional_accuracy,
            "baseline_brier_up": baseline_brier_up,
            "baseline_reference_directional_accuracy": baseline_reference_accuracy,
            "baseline_reference_brier_up": baseline_reference_brier,
            "baseline_mae_return": baseline_mae_return,
            "baseline_models": baseline_models,
            "early_stopping_patience": int(early_stopping_patience),
            "seed": int(seed) if seed is not None else None,
            "epoch_metrics": epoch_history,
            "loss_type": "nll" if use_nll_loss else "mse",
            "class_count": class_count,
            "feature_count": feature_dim,
            "feature_names": feature_names,
        }

        self._artifact = TrainedModelArtifact(
            version=self.trained_version,
            weights=weights,
            metadata={
                "epochs": epochs,
                "batch_size": batch_size,
                "batches_per_epoch": num_batches,
                "samples": total_samples,
                "kca_heads": int(kca_heads),
                "kca_state_dim": int(kca_state_dim),
                "kca_num_layers": int(kca_num_layers),
                "kca_hidden_dim": int(kca_hidden_dim),
                "kca_slow_stride": int(kca_slow_stride),
                "horizon": horizon,
                "instrument": getattr(instrument, "symbol", None) if instrument else None,
                "target_mode": "net_of_fee" if net_target else "raw_return",
                "estimated_fee_bps": float(estimated_fee_bps),
                "val_gate_min_directional_accuracy": float(min_val_directional_accuracy),
                "val_gate_min_baseline_improvement": float(min_baseline_improvement),
                "val_gate_brier_improvement_margin": float(brier_improvement_margin),
                "baseline_reference_directional_accuracy": baseline_reference_accuracy,
                "baseline_reference_brier_up": baseline_reference_brier,
                "deployable": bool(deployable),
                "seed": int(seed) if seed is not None else None,
                "class_count": class_count,
                "feature_names": feature_names,
            },
        )

        self.training_history.append(result)
        return result

    def export_weights(self) -> bytes:
        """
        Serialize model weights for distribution.
        
        Returns:
            bytes in format:
            - First 8 bytes: version string length (big-endian uint64)
            - Next N bytes: version string (UTF-8)
            - Remaining: pickled model_weights dict
        
        This format is consumed by KCAPredictor.request_model_update()
        
        ⚠️ Production Risk: pickle is not secure and not language-agnostic.
        For production systems, consider:
        - torch.save() for PyTorch models
        - joblib for scikit-learn
        - msgpack + numpy for cross-language compatibility
        - protobuf for strict versioning
        """
        if self._artifact is None:
            raise RuntimeError("No trained weights available")
        
        # Encode version
        version_bytes = self._artifact.version.encode("utf-8")
        version_length = len(version_bytes).to_bytes(8, byteorder="big")
        
        # Serialise model weights via torch (avoids bare pickle)
        _buf = io.BytesIO()
        torch.save(self._artifact.weights, _buf)
        weights_payload = _buf.getvalue()
        
        # Combine
        return version_length + version_bytes + weights_payload

    async def push_model(self, update_port: IModelUpdatePort) -> None:
        """
        Push trained model to production via ModelUpdatePort.
        
        The production system predictor receives update via request_model_update().
        
        Args:
            update_port: IModelUpdatePort implementer (e.g., KCAPredictor)
        """
        if self._artifact is None:
            raise RuntimeError("No trained weights available")
        weights_bytes = self.export_weights()
        await update_port.request_model_update(weights_bytes, self._artifact.version)

    def get_training_history(self) -> list[dict[str, Any]]:
        """Return list of training runs."""
        return list(self.training_history)

    def get_latest_artifact(self) -> TrainedModelArtifact | None:
        """Return latest trained model artifact, if available."""
        return self._artifact

