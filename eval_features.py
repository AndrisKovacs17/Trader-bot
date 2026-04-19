"""
Quick evaluation: train TinyGRU on the FULL 33-feature set and compare to baseline.
Uses 90-day real BTC data, same setup as the ablation study.
"""
from __future__ import annotations
import math, random, sys, time
import urllib.request, json

# -- Hyper-params -------------------------------------------------------------
LOOKBACK  = 32
HORIZON   = 6
EPOCHS    = 20
BATCH     = 128
LR        = 3e-3
SEED      = 42

# -- Tiny GRU model (pure Python/math, no PyTorch needed here — but we use torch) --
try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("PyTorch required")

random.seed(SEED); torch.manual_seed(SEED)

# -- Fetch real BTC data -------------------------------------------------------
def fetch_btc(days: int = 90) -> list[dict]:
    bars = []
    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    limit    = 1000
    t        = start_ms
    while t < end_ms:
        url = (
            "https://api.binance.com/api/v3/klines"
            f"?symbol=BTCUSDT&interval=5m&startTime={t}&limit={limit}"
        )
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.loads(r.read())
        if not data:
            break
        for k in data:
            bars.append({
                "ts": k[0], "open": float(k[1]), "high": float(k[2]),
                "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]),
                "taker_buy_volume": float(k[9]),
            })
        t = data[-1][0] + 1
        time.sleep(0.05)
    print(f"Fetched {len(bars)} bars")
    return bars

# -- Feature engineering (uses the real codebase) -----------------------------
sys.path.insert(0, "/workspace/diplomamunkakod")
from core.ml.feature_engineering import build_trade_feature_rows, FEATURE_NAMES

def make_dataset(bars: list[dict]):
    rows, names = build_trade_feature_rows(bars)
    assert names == FEATURE_NAMES, f"Feature mismatch: {names} vs {FEATURE_NAMES}"
    print(f"Features ({len(names)}): {names}")
    return rows, names

# -- Label generation ---------------------------------------------------------
def make_labels(bars: list[dict], horizon: int, epsilon: float = 5e-5):
    prices = [b["close"] for b in bars]
    labels = []
    for i in range(len(prices)):
        if i + horizon < len(prices):
            fwd = (prices[i + horizon] - prices[i]) / max(prices[i], 1e-8)
            labels.append(1 if fwd > epsilon else 0)
        else:
            labels.append(-1)   # invalid
    return labels

# -- Sequence builder ---------------------------------------------------------
def make_sequences(rows, labels, lookback):
    X, Y = [], []
    for i in range(lookback, len(rows)):
        if labels[i] == -1:
            continue
        X.append(rows[i - lookback: i])
        Y.append(labels[i])
    return X, Y

# -- TinyGRU ------------------------------------------------------------------
class TinyGRU(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.gru  = nn.GRU(n_feat, 32, batch_first=True)
        self.head = nn.Linear(32, 2)
    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h.squeeze(0))

# -- Training loop -------------------------------------------------------------
def train_eval(X, Y, n_feat, tag="model"):
    n = len(X)
    split = int(n * 0.8)
    # --- class balance check ---
    y_train = Y[:split]
    pos = sum(y_train) / len(y_train)
    print(f"  Train UP ratio: {pos:.3f}  (val: {sum(Y[split:])/len(Y[split:]):.3f})")
    
    # tensors
    Xt = torch.tensor(X[:split],  dtype=torch.float32)
    Yt = torch.tensor(Y[:split],  dtype=torch.long)
    Xv = torch.tensor(X[split:],  dtype=torch.float32)
    Yv = torch.tensor(Y[split:],  dtype=torch.long)

    # z-score normalize (fit on train only)
    flat = Xt.view(-1, n_feat)
    mean = flat.mean(0); std = flat.std(0).clamp_min(1e-6)
    Xt = (Xt - mean) / std
    Xv = (Xv - mean) / std
    
    model = TinyGRU(n_feat)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    
    idx = list(range(len(Xt)))
    best_val_acc = 0.0
    for ep in range(1, EPOCHS + 1):
        model.train()
        random.shuffle(idx)
        total_loss = 0.0
        for b in range(0, len(idx), BATCH):
            bi = idx[b: b + BATCH]
            xb = Xt[bi]; yb = Yt[bi]
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(bi)
        
        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_preds  = val_logits.argmax(1)
            val_acc    = (val_preds == Yv).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        avg_loss = total_loss / len(idx)
        print(f"  [{tag}] ep {ep:02d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}  best={best_val_acc:.4f}")
    
    return best_val_acc

# -- Baseline: random / majority -----------------------------------------------
def baseline_majority(Y, split_ratio=0.8):
    n = len(Y)
    split = int(n * split_ratio)
    y_val = Y[split:]
    majority = int(sum(y_val) >= len(y_val) / 2)
    acc = sum(1 for y in y_val if y == majority) / len(y_val)
    return acc

# -- Main ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Fetching 90-day BTCUSDT 5m data...")
    bars = fetch_btc(days=90)
    
    print("\nBuilding features...")
    rows, names = make_dataset(bars)
    n_feat = len(names)
    
    labels = make_labels(bars, HORIZON)
    X, Y = make_sequences(rows, labels, LOOKBACK)
    print(f"Sequences: {len(X)}  (train={int(len(X)*0.8)}, val={len(X)-int(len(X)*0.8)})")
    
    majority_acc = baseline_majority(Y)
    print(f"\nMajority baseline acc: {majority_acc:.4f}")
    
    print(f"\nTraining TinyGRU on ALL {n_feat} features...")
    t0 = time.time()
    best = train_eval(X, Y, n_feat, tag="full33")
    elapsed = time.time() - t0
    
    print("\n" + "=" * 60)
    print(f"RESULT — {n_feat} features")
    print(f"  Majority baseline:  {majority_acc:.4f}")
    print(f"  Best val accuracy:  {best:.4f}  (+{best - majority_acc:+.4f} vs majority)")
    print(f"  Training time:      {elapsed:.1f}s")
    print("=" * 60)
