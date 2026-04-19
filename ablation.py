"""
Feature ablation study — runs on REAL BTCUSDT 5m Binance data.

Fetches ~90 days of 5m bars (~26k bars) via the public Binance API (no API key needed).
Tests cumulative feature groups; uses a tiny GRU model for speed.

Run: python3 ablation.py
"""
from __future__ import annotations
import math, random, sys, time
random.seed(42)

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch not found. Install torch to run ablation.")
    sys.exit(1)

torch.manual_seed(42)


# --- real Binance data fetch --------------------------------------------------
def fetch_btc_bars(days: int = 90) -> list[dict]:
    """Fetch BTCUSDT 5m klines from Binance public API. No API key required."""
    sys.path.insert(0, "/workspace/diplomamunkakod")
    from adapters.infrastructure.binance_feed import get_historical_klines_with_taker
    print(f"Fetching BTCUSDT 5m klines ({days} days)...")
    df = get_historical_klines_with_taker(
        symbol="BTCUSDT",
        interval="5m",
        start_str=f"{days} days ago UTC",
        max_bars=50000,
    )
    if df.empty:
        print("ERROR: No klines returned. Check internet connection.")
        sys.exit(1)
    bars = []
    for _, row in df.iterrows():
        ts_val = row["T"]
        ts_ms = int(ts_val.value // 1_000_000) if hasattr(ts_val, "value") else int(ts_val.timestamp() * 1000)
        bars.append({
            "price":        float(row["p"]),
            "open":         float(row.get("o", row["p"])),
            "high":         float(row.get("h", row["p"])),
            "low":          float(row.get("l", row["p"])),
            "volume":       float(row["q"]),
            "taker_buy_vol": float(row.get("taker_buy_vol", -1.0)),
            "ts_ms":        ts_ms,
            "funding_rate": 0.0,
        })
    print(f"  Got {len(bars)} bars.")
    return bars


# --- feature groups -----------------------------------------------------------
# These map to indices in FEATURE_NAMES from feature_engineering.py
# Group ablation: cumulative (each group adds on top of previous)
FEATURE_GROUPS: list[tuple[str, list[str]]] = [
    ("baseline_ret",    ["ret"]),
    ("+momentum",       ["ret_mean_8", "ret_mean_32", "ret_z_32", "trend_quality_8", "streak_8"]),
    ("+volatility",     ["ret_vol_8", "ret_vol_32", "vol_z_16", "atr_14_z", "vol_ratio_log"]),
    ("+order_flow",     ["taker_vol_ratio", "bid_ask_proxy", "volume_accel", "log_volume", "ofi_delta"]),
    ("+technicals",     ["rsi_14", "bb_pos_20", "macd_norm", "stoch_k", "breakout_32"]),
    ("+structure",      ["dist_ma_16", "dist_ma_64", "dist_ma_128", "trend_ma_8_32"]),
    ("+microstructure", ["vwap_dist_64", "hurst_rs_32", "ret_autocorr", "vol_ret_align"]),
    ("+candles",        ["body_ratio", "upper_shadow", "lower_shadow", "engulfing", "dist_high_64", "dist_low_64"]),
    ("+time",           ["intraday_sin", "intraday_cos", "weekday_sin", "weekday_cos"]),
    ("+funding",        ["funding_rate_z"]),
]


# --- tiny GRU model for speed -------------------------------------------------
class TinyGRU(nn.Module):
    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(feat_dim, 32, batch_first=True)
        self.head = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(h[-1])


def run_ablation(
    feat_rows: list[list[float]],
    feat_names: list[str],
    targets: list[int],  # binary 0/1
    active_feats: list[str],
    lookback: int = 32,
    horizon: int = 6,
    epochs: int = 15,
    batch_size: int = 128,
) -> float:
    """Train TinyGRU on active features, return val_acc."""
    feat_idx = {n: i for i, n in enumerate(feat_names)}
    cols = [feat_idx[f] for f in active_feats if f in feat_idx]
    if not cols:
        return float("nan")

    n = len(feat_rows) - lookback - horizon
    if n < 200:
        return float("nan")

    x_data = torch.tensor([[feat_rows[i + t][c] for c in cols] for i in range(n) for t in range(lookback)],
                          dtype=torch.float32).view(n, lookback, len(cols))
    y_data = torch.tensor(targets[lookback: lookback + n], dtype=torch.long)

    # Normalize
    flat = x_data.view(-1, len(cols))
    mean = flat.mean(0); std = flat.std(0).clamp_min(1e-6)
    x_data = (x_data - mean) / std

    split = int(n * 0.8)
    x_tr, y_tr = x_data[:split], y_data[:split]
    x_va, y_va = x_data[split:], y_data[split:]

    model = TinyGRU(len(cols))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    ce = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), batch_size):
            idx = perm[i:i+batch_size]
            loss = ce(model(x_tr[idx]), y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(x_va).argmax(1)
        acc = float((preds == y_va).float().mean())
    return acc


def main() -> None:
    bars = fetch_btc_bars(days=90)

    from core.ml.feature_engineering import build_trade_feature_rows, FEATURE_NAMES
    feat_rows, feat_names = build_trade_feature_rows(bars)

    # Binary targets: UP=1 if next-horizon net return > 0
    fee = 10.0 / 10_000 * 2  # roundtrip 10bps
    horizon = 6
    prices = [b["price"] for b in bars]
    targets = []
    for i in range(len(prices)):
        if i + horizon < len(prices):
            gross = (prices[i + horizon] - prices[i]) / max(prices[i], 1e-8)
            net = gross - fee if gross > 0 else gross + fee
            targets.append(1 if net > 0 else 0)
        else:
            targets.append(0)

    up_frac = sum(targets) / len(targets)
    print(f"Dataset: {len(feat_rows)} rows, UP={up_frac:.2f}, DOWN={1-up_frac:.2f}")
    print(f"\n{'-'*62}")
    print(f"  {'Group':<22}  {'Feats':>5}  {'ValAcc':>7}  {'Delta':>8}  {'Verdict'}")
    print(f"{'-'*62}")

    active: list[str] = []
    prev_acc: float | None = None
    results: list[tuple[str, list[str], float]] = []

    for group_name, new_feats in FEATURE_GROUPS:
        active = active + [f for f in new_feats if f in {n: 1 for n in FEATURE_NAMES}]
        t0 = time.time()
        acc = run_ablation(feat_rows, feat_names, targets, active, lookback=32, horizon=horizon)
        elapsed = time.time() - t0

        if prev_acc is None:
            delta_str = "     ---"
            verdict = ""
        else:
            d = acc - prev_acc
            delta_str = f"{d:>+8.4f}"
            verdict = "[OK] HELPS" if d > 0.003 else ("[X] HURTS" if d < -0.003 else "~ neutral")

        print(f"  {group_name:<22}  {len(active):>5}  {acc:>7.4f}  {delta_str}  {verdict}  ({elapsed:.1f}s)")
        results.append((group_name, new_feats, acc))
        prev_acc = acc

    print(f"{'-'*62}")

    # Individual feature contribution for groups that *hurt*
    hurting = [(g, feats) for g, feats, acc in results for g2, feats2, acc2 in results
               if g == g2 and acc2 < prev_acc - 0.003]  # type: ignore[assignment]
    # Compute per-feature delta within hurting groups
    print("\n--- Individual feature scan in groups that changed accuracy ---")
    affected_groups = [(g, feats, acc) for g, feats, acc in results
                       if len(results) > 1 and abs(acc - results[results.index((g,feats,acc)) - 1][2]) > 0.003]
    if not affected_groups:
        print("  No strongly affected groups. All groups within ±0.003.")
    else:
        for g_name, g_feats, g_acc in affected_groups:
            print(f"\n  Scanning individual features in {g_name}:")
            # Build base = all features BEFORE this group
            idx = [i for i, (n, _, _) in enumerate(results) if n == g_name][0]
            base_feats: list[str] = []
            for prev_g, prev_f, _ in results[:idx]:
                base_feats.extend(prev_f)
            base_acc = results[idx - 1][2] if idx > 0 else 0.5
            for feat in g_feats:
                test_feats = base_feats + [feat]
                a = run_ablation(feat_rows, feat_names, targets, test_feats, lookback=32, horizon=horizon, epochs=10)
                d = a - base_acc
                print(f"    {feat:<28}  acc={a:.4f}  delta={d:+.4f}  {'HELPS' if d > 0.003 else ('HURTS' if d < -0.003 else 'neutral')}")


if __name__ == "__main__":
    main()
