"""Hyperparameter sweep for the KCA-Mamba offline training engine.

Runs the offline training pipeline with several preset hyperparameter
configurations on the same historical Binance data, then keeps the model
that scores best on a composite of validation directional accuracy and
Brier score. The winner's weights are persisted via `core.ml.model_store`
so that subsequent `python main.py` runs load it instead of retraining.

Usage:
    python -m tools.hp_sweep                # run default presets
    python -m tools.hp_sweep --presets fast # run only the 'fast' subset (debug)
    python -m tools.hp_sweep --days 90      # override training window

Composite score (higher is better):
    score = val_directional_accuracy
            - 0.5 * val_brier_up
            + 0.25 * (val_directional_accuracy - baseline_reference_acc)

The script never auto-deploys a non-deployable model: only winners that
the in-engine validation gate marked `deployable=True` are saved to disk.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any

# Allow running as `python tools/hp_sweep.py` from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.infrastructure.binance_feed import (
    get_historical_klines_with_taker,
    get_historical_funding_rates,
)
from adapters.offline_training.training_engine import TrainingEngine
from core.domain.models import Instrument
from core.ml import model_store


# Default hyperparam space. Each preset overlays on top of `_BASE`.
_BASE: dict[str, Any] = {
    "epochs": 40,
    "batch_size": 128,
    "hidden_size": 64,
    "num_layers": 2,
    "learning_rate": 3e-4,
    "use_nll_loss": True,
    "lookback": 64,
    "horizon": 6,
    "kca_heads": 4,
    "kca_state_dim": 16,
    "kca_num_layers": 1,
    "kca_hidden_dim": 64,
    "kca_slow_stride": 8,
    "direction_epsilon": 5e-5,
    "balance_direction_loss": True,
    "direction_pos_weight_min": 0.5,
    "direction_pos_weight_max": 3.0,
    "cls_logit_l2": 0.0,
    "mu_l2_reg": 0.0,
    "estimated_fee_bps": 5.0,
    "net_target": False,
    "seed": 42,
    "early_stopping_patience": 10,
    "min_val_directional_accuracy": 0.505,
    "min_baseline_improvement": 0.0,
    "brier_improvement_margin": -0.01,
    "collapse_max_class_share": 0.90,
    "collapse_min_side_share": 0.10,
    "run_simple_baselines": True,
    "baseline_epochs": 2,
    "cls_loss_weight": 3.0,
    "nll_loss_clip": -8.0,
    "head_dropout": 0.15,
    "weight_decay": 0.01,
    "label_smoothing": 0.0,
    "sequence_stride": 4,
}

PRESETS: dict[str, dict[str, Any]] = {
    "baseline":           {},
    "deeper_state":       {"kca_num_layers": 2, "kca_state_dim": 24},
    "label_smooth":       {"label_smoothing": 0.05},
    "denser_seqs":        {"sequence_stride": 2},
    "lower_lr":           {"learning_rate": 1e-4, "early_stopping_patience": 12},
    "stronger_drop":      {"head_dropout": 0.25, "weight_decay": 0.02},
    "more_heads":         {"kca_heads": 8, "kca_state_dim": 24},
    "longer_lookback":    {"lookback": 96, "kca_slow_stride": 12},
}

FAST_SUBSET = {"baseline", "label_smooth", "denser_seqs"}


@dataclass
class SweepResult:
    name: str
    overrides: dict[str, Any]
    result: dict[str, Any] = field(default_factory=dict)
    weights: bytes | None = None
    score: float = float("-inf")
    error: str | None = None


def _composite_score(r: dict[str, Any]) -> float:
    acc = float(r.get("val_directional_accuracy") or float("nan"))
    brier = float(r.get("val_brier_up") or float("nan"))
    base_acc = float(r.get("baseline_reference_directional_accuracy") or float("nan"))
    if math.isnan(acc) or math.isnan(brier):
        return float("-inf")
    gain = (acc - base_acc) if not math.isnan(base_acc) else 0.0
    return acc - 0.5 * brier + 0.25 * gain


def build_dataset(days: int, interval: str = "5m", max_bars: int = 110000) -> list[dict[str, Any]]:
    print(f"[SWEEP] Fetching {days}d of {interval} klines (max_bars={max_bars})...")
    df = get_historical_klines_with_taker(
        symbol="BTCUSDT",
        interval=interval,
        start_str=f"{days} days ago UTC",
        max_bars=max_bars,
    )
    if df.empty:
        raise RuntimeError("Empty kline dataframe from Binance")
    df = df.sort_values("T").reset_index(drop=True)
    df["ret"] = df["p"].pct_change().fillna(0.0)

    print("[SWEEP] Fetching funding rates...")
    funding_map = get_historical_funding_rates(
        symbol="BTCUSDT",
        start_str=f"{days} days ago UTC",
    )

    def _nearest_funding(ts_ms: int) -> float:
        if not funding_map:
            return 0.0
        best_ts = max((t for t in funding_map if t <= ts_ms), default=None)
        return funding_map[best_ts] if best_ts is not None else 0.0

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        ts_ms = int(getattr(row["T"], "value", 0) // 1_000_000)
        rows.append({
            "price":  float(row["p"]),
            "open":   float(row.get("o", row["p"])),
            "high":   float(row.get("h", row["p"])),
            "low":    float(row.get("l", row["p"])),
            "return": float(row["ret"]),
            "volume": float(row["q"]),
            "ts_ms":  ts_ms,
            "taker_buy_vol": float(row.get("taker_buy_vol", -1.0)),
            "funding_rate": _nearest_funding(ts_ms),
        })
    print(f"[SWEEP] Built dataset with {len(rows)} rows")
    return rows


def run_one(name: str, overrides: dict[str, Any], dataset: list[dict[str, Any]]) -> SweepResult:
    cfg = {**_BASE, **overrides}
    print(f"\n{'='*72}\n[SWEEP] Preset: {name}")
    print(f"        Overrides: {overrides}\n{'='*72}")
    engine = TrainingEngine()
    sr = SweepResult(name=name, overrides=overrides)
    try:
        result = engine.train(
            dataset=dataset,
            instrument=Instrument(symbol="BTCUSDT"),
            **cfg,
        )
        sr.result = result
        sr.score = _composite_score(result)
        if result.get("deployable") and engine._artifact is not None:
            sr.weights = engine.export_weights()
    except Exception as exc:  # noqa: BLE001
        sr.error = str(exc)
        print(f"[SWEEP] Preset {name} FAILED: {exc}")
    return sr


def main() -> int:
    parser = argparse.ArgumentParser(description="KCA-Mamba hyperparameter sweep")
    parser.add_argument("--days", type=int, default=365, help="History window size in days")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--max-bars", type=int, default=110000)
    parser.add_argument("--presets", choices=("all", "fast"), default="all")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save the best deployable model as the active one (default: on)")
    parser.add_argument("--no-save", dest="save", action="store_false")
    args = parser.parse_args()

    dataset = build_dataset(args.days, args.interval, args.max_bars)

    chosen = list(PRESETS.items())
    if args.presets == "fast":
        chosen = [(n, o) for n, o in chosen if n in FAST_SUBSET]

    results: list[SweepResult] = []
    for name, overrides in chosen:
        sr = run_one(name, overrides, dataset)
        results.append(sr)

    # Print summary
    print("\n" + "=" * 72)
    print(f"{'preset':<20} {'val_acc':>8} {'brier':>8} {'gain':>8} {'score':>8} {'deploy':>7}")
    print("-" * 72)
    for sr in sorted(results, key=lambda r: r.score, reverse=True):
        r = sr.result
        if sr.error:
            print(f"{sr.name:<20} ERROR: {sr.error}")
            continue
        acc = r.get("val_directional_accuracy")
        brier = r.get("val_brier_up")
        base = r.get("baseline_reference_directional_accuracy")
        gain = (acc - base) if (acc is not None and base is not None) else None
        deploy = "YES" if r.get("deployable") else "no"
        print(
            f"{sr.name:<20} "
            f"{acc:>8.4f} {brier:>8.4f} "
            f"{(gain if gain is not None else 0.0):>+8.4f} "
            f"{sr.score:>8.4f} {deploy:>7}"
        )
    print("=" * 72)

    deployable = [sr for sr in results if sr.weights is not None]
    if not deployable:
        print("\n[SWEEP] No preset produced a deployable model. Nothing saved.")
        return 1

    winner = max(deployable, key=lambda r: r.score)
    print(f"\n[SWEEP] Winner: {winner.name}  score={winner.score:.4f}  "
          f"acc={winner.result.get('val_directional_accuracy'):.4f}")

    if args.save:
        meta = {
            "preset": winner.name,
            "overrides": winner.overrides,
            "val_directional_accuracy": winner.result.get("val_directional_accuracy"),
            "val_brier_up": winner.result.get("val_brier_up"),
            "baseline_reference_directional_accuracy":
                winner.result.get("baseline_reference_directional_accuracy"),
            "best_epoch": winner.result.get("best_epoch"),
            "horizon": winner.result.get("horizon"),
            "lookback": winner.result.get("lookback") or _BASE["lookback"],
            "feature_count": winner.result.get("feature_count"),
            "dataset_size": len(dataset),
            "deploy_reason": winner.result.get("deploy_reason"),
            "data_source": f"binance-{args.interval}-{args.days}d",
            "sweep_score": winner.score,
        }
        bin_path, meta_path = model_store.save_active(
            winner.weights,  # type: ignore[arg-type]
            version=str(winner.result.get("version") or "sweep-winner"),
            metadata=meta,
        )
        print(f"[SWEEP] Saved active model: {bin_path}")
        print(f"[SWEEP] Metadata sidecar:   {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
