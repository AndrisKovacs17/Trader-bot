"""LTSF Benchmark: VanillaLSTM vs FairMamba vs KCA-Mamba on real LTSF datasets.

Datasets (Exchange / Weather / ECL) via HuggingFace. 70/10/20 train-val-test split.
"""
from __future__ import annotations

import copy
import io
import os
import pathlib
import threading
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any

# ── Shared state ──────────────────────────────────────────────────────────────
_lock: threading.Lock = threading.Lock()
_state: dict[str, Any] = {"status": "idle"}

# ── Defaults (matching the notebook) ─────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "dataset":    "Exchange",
    "seq_len":    96,
    "stride":     1,      # window stride (1=max overlap; növeld csak nagy adathalmaznál)
    "batch_size": 64,
    "epochs":     5,
    "lr":         1e-3,
    "lstm_hidden":  128,
    "fair_heads":   4,
    "fair_state":   32,
    "kca_heads":    4,
    "kca_state":    32,
    "noise_scale": 0.0,   # extra Gaussian noise added to train (0 = off)
    "arima_p":      5,    # AR order for the ARIMA baseline
    "arima_q":      3,    # MA order for the ARIMA baseline
}

_URLS: dict[str, str] = {
    "Exchange": "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/exchange_rate.csv",
    "Weather":  "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/weather.csv",
    "ECL":      "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/electricity.csv",
    # ETT (Electricity Transformer Temperature) — Zeng et al. 2023 standard LTSF benchmarks
    "ETTh1":    "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTh1.csv",
    "ETTh2":    "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTh2.csv",
    "ETTm1":    "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTm1.csv",
    "ETTm2":    "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTm2.csv",
    # M4 Hourly — 414 óránkénti sorozat, változó hossz (700–1000 lépés/sor)
    # Formátum: soronként 1 sorozat (V1=ID, V2..VN=értékek, NaN-paddelt)
    # Összefűzve egyetlen hosszú 1D sorozattá → 1-step autoregresszív feladat
    "M4Hourly": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Hourly-train.csv",
}

# Fallback URL-ek ha a HuggingFace nem elérhető (thuml/Time-Series-Library eredeti forrás)
_FALLBACK_URLS: dict[str, str] = {
    "Weather": "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/weather/weather.csv",
    "ETTh1":   "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/ETT-small/ETTh1.csv",
    "ETTh2":   "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/ETT-small/ETTh2.csv",
    "ETTm1":   "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/ETT-small/ETTm1.csv",
    "ETTm2":   "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/ETT-small/ETTm2.csv",
    "ECL":     "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/electricity/electricity.csv",
    "Exchange":"https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/exchange_rate/exchange_rate.csv",
}


def get_state() -> dict:
    with _lock:
        return copy.deepcopy(_state)


def trigger(cfg: dict | None = None) -> bool:
    """Start the benchmark in a background thread. Returns False if already running."""
    with _lock:
        if _state.get("status") == "running":
            return False
        _state.clear()
        _state["status"] = "running"
        _state["started_at"] = _now()
    t = threading.Thread(target=_run, kwargs={"cfg": cfg or {}}, daemon=True)
    t.start()
    return True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_list(t) -> list:
    import math
    return [v if math.isfinite(v) else 0.0 for v in t.detach().cpu().float().numpy().tolist()]


# ── Dataset cache ─────────────────────────────────────────────────────────────
# CSV-fájlok helyi másolatát ide menti. Ha nincs internet, innen tölt be.
_CACHE_DIR = pathlib.Path(__file__).parent / "dataset_cache"


def _fetch_raw(name: str) -> str:
    """Return raw CSV text for *name*. Downloads once and caches locally.
    Tries primary URL first, then fallback URL, then local cache."""
    _CACHE_DIR.mkdir(exist_ok=True)
    cache_file = _CACHE_DIR / f"{name}.csv"
    urls_to_try = [u for u in [_URLS.get(name), _FALLBACK_URLS.get(name)] if u]
    # ── Try each URL ─────────────────────────────────────────────────────────
    for url in urls_to_try:
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
            cache_file.write_text(raw, encoding="utf-8")
            return raw
        except Exception:
            continue
    # ── Offline fallback ──────────────────────────────────────────────────────
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    raise RuntimeError(
        f"Nem sikerült letölteni '{name}' adathalmazt és nincs lokális cache sem. "
        f"Futtasd egyszer internetkapcsolattal: {urls_to_try[0] if urls_to_try else '?'}"
    )



def _load_dataset(name: str, seq_len: int, batch_size: int, noise_scale: float = 0.0, stride: int = 24):
    import pandas as pd
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader

    class _Scaler:
        def fit_transform(self, X: "np.ndarray") -> "np.ndarray":
            self.mean_ = X.mean(axis=0)
            self.std_  = X.std(axis=0) + 1e-8
            return ((X - self.mean_) / self.std_).astype("float32")
        def transform(self, X: "np.ndarray") -> "np.ndarray":
            return ((X - self.mean_) / self.std_).astype("float32")

    # ── M4 Hourly speciális betöltő ──────────────────────────────────────────
    # Formátum: soronként 1 sorozat (V1=ID, V2..VN=értékek, NaN-paddelt).
    # Minden sorozatot NaN nélkül kicsipünk, majd összefűzzük egyetlen 1D
    # tömbé → (N_total, 1) alakban adjuk tovább, mint egydimenzós idősor.
    if name == "M4Hourly":
        raw = _fetch_raw("M4Hourly")
        df = pd.read_csv(io.StringIO(raw))
        # Első oszlop az ID (V1), a többi az értékek
        val_cols = [c for c in df.columns if c != df.columns[0]]
        series_list = []
        for _, row in df[val_cols].iterrows():
            vals = row.dropna().values.astype("float32")
            if len(vals) > seq_len + 1:
                series_list.append(vals)
        # Összefűzés: minden sorozat külön szegmensként, közöttük nincs folytonosság
        # → de a Dataset csak seq_len ablakokat vesz, határsértés nem fordul elő
        data = np.concatenate(series_list).reshape(-1, 1)  # (N_total, 1)
    else:
        raw = _fetch_raw(name)
        df = pd.read_csv(io.StringIO(raw))
        if "date" in df.columns:
            df = df.drop(columns=["date"])
        data = df.values.astype("float32")

    n = len(data)
    train_sz = int(n * 0.7)
    val_sz   = int(n * 0.1)

    scaler = _Scaler()
    train_clean = scaler.fit_transform(data[:train_sz])
    test_data   = scaler.transform(data[train_sz + val_sz:])

    # Noisy train input: x is corrupted, y stays clean.
    # Pre-compute noise once per run at init time (fast: no per-sample allocation).
    # Clipped to ±3σ to prevent extreme outliers blowing up training at high noise_scale.
    _noise_scale = float(noise_scale)
    if _noise_scale > 0.0:
        rng = np.random.default_rng()   # different seed every run
        # train noise
        tr_noise = rng.standard_normal(train_clean.shape).astype("float32") * _noise_scale
        tr_noise = np.clip(tr_noise, -3.0 * _noise_scale, 3.0 * _noise_scale)
        train_noisy = train_clean + tr_noise
        # test noise — same scale so model sees same distribution
        te_noise = rng.standard_normal(test_data.shape).astype("float32") * _noise_scale
        te_noise = np.clip(te_noise, -3.0 * _noise_scale, 3.0 * _noise_scale)
        test_noisy = test_data + te_noise
    else:
        train_noisy = train_clean
        test_noisy  = test_data

    _stride = max(1, int(stride))

    class _DS(Dataset):
        def __init__(self, inp, tgt):
            # inp: what the model sees (may be noisy), tgt: what the model must predict (always clean)
            self.inp = inp
            self.tgt = tgt
            # Pre-compute valid start indices with given stride → less correlated gradient updates
            self.indices = list(range(0, max(0, len(tgt) - seq_len), _stride))
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            s = self.indices[i]
            x = torch.from_numpy(self.inp[s     : s + seq_len])
            y = torch.from_numpy(self.tgt[s + 1 : s + seq_len + 1])
            return x, y

    train_loader = DataLoader(_DS(train_noisy, train_clean), batch_size=batch_size, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(_DS(test_noisy,  test_data),   batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader, int(data.shape[1])


# ── Model factories ───────────────────────────────────────────────────────────

def _make_lstm(input_dim: int, hidden_dim: int = 128):
    import torch.nn as nn

    class VanillaLSTM(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.lstm     = nn.LSTM(d, h, 1, batch_first=True)
            self.out_proj = nn.Linear(h, d)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.out_proj(out)

    return VanillaLSTM(input_dim, hidden_dim)


def _make_fair_mamba(d_model: int, heads: int, d_state: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FairMambaBlock(nn.Module):
        def __init__(self, d, h, s, ks=4):
            super().__init__()
            self.H, self.D, self.E = h, s, h * s
            self.in_proj  = nn.Linear(d, self.E * 2)
            self.conv1d   = nn.Conv1d(self.E, self.E, ks, groups=self.E, padding=ks-1)
            self.dt_net   = nn.Sequential(nn.Linear(self.E, 64), nn.SiLU(), nn.Linear(64, self.E))
            self.proj_B   = nn.Linear(self.E, self.E)
            self.proj_C   = nn.Linear(self.E, self.E)
            self.out_proj = nn.Linear(self.E, d)

        def parallel_scan(self, A, B, h0):
            T, step = A.shape[1], 1
            while step < T:
                Ar, As, Bs = A[:, step:], A[:, :-step], B[:, :-step]
                A = torch.cat([A[:, :step], Ar * As],          dim=1)
                B = torch.cat([B[:, :step], B[:, step:] + Ar * Bs], dim=1)
                step *= 2
            return B + A * h0.unsqueeze(1)

        def forward(self, x):
            B, T, _ = x.shape
            xw, xg = self.in_proj(x).chunk(2, dim=-1)
            gate   = F.silu(xg)
            xc     = F.silu(self.conv1d(xw.transpose(1, 2))[:, :, :T].transpose(1, 2))
            dt     = F.softplus(self.dt_net(xc)).clamp(1e-4, 0.1)
            h0     = torch.zeros(B, self.E, device=x.device)
            h      = self.parallel_scan(torch.exp(-dt), dt * self.proj_B(xc), h0)
            return x + self.out_proj(self.proj_C(xc) * h * gate)

    return FairMambaBlock(d_model, heads, d_state)


def _make_kca_mamba(d_model: int, heads: int, d_state: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class KCAMambaBlock(nn.Module):
        def __init__(self, d, h, s, ks=4):
            super().__init__()
            self.H, self.D, self.E = h, s, h * s
            self.in_proj  = nn.Linear(d, self.E * 2)
            self.conv1d   = nn.Conv1d(self.E, self.E, ks, groups=self.E, padding=ks-1)
            self.out_proj = nn.Linear(self.E, d)
            self.res_proj = nn.Linear(d, d)
            self.proj_v   = nn.Linear(self.E, self.E)
            self.Q_net = nn.Sequential(nn.Linear(self.E, 64), nn.SiLU(), nn.Linear(64, h))
            self.R_net = nn.Sequential(nn.Linear(self.E, 64), nn.SiLU(), nn.Linear(64, h))
            self.K_net = nn.Sequential(nn.Linear(self.E, 64), nn.SiLU(), nn.Linear(64, h))
            # Globális skálár: K_base≈0.068 → ~14 lépés init, a modell maga tanulja a memóriát
            self.q_scale = nn.Parameter(torch.tensor(-2.0))
            self.r_scale = nn.Parameter(torch.tensor( 1.5))
            self.mu_init      = nn.Parameter(torch.zeros(1, h, s))
            self.res_gate_bias = nn.Parameter(torch.tensor(-1.0))
            self.v_norm       = nn.LayerNorm(self.E)

        def parallel_scan(self, A, B, mu0):
            T, step = A.shape[1], 1
            while step < T:
                Ar, As, Bs = A[:, step:], A[:, :-step], B[:, :-step]
                A = torch.cat([A[:, :step], Ar * As],          dim=1)
                B = torch.cat([B[:, :step], B[:, step:] + Ar * Bs], dim=1)
                step *= 2
            return B + A * mu0.unsqueeze(1)

        def forward(self, x):
            B, T, _ = x.shape
            dtype = x.dtype
            xw, xg = self.in_proj(x).chunk(2, dim=-1)
            gate   = F.silu(xg)
            xc     = F.silu(self.conv1d(xw.transpose(1, 2))[:, :, :T].transpose(1, 2))
            v      = self.v_norm(self.proj_v(xc)).view(B, T, self.H, self.D)
            Q = (F.softplus(self.Q_net(xc)) * F.softplus(self.q_scale)).unsqueeze(-1)
            R = (F.softplus(self.R_net(xc)) * F.softplus(self.r_scale)).unsqueeze(-1)
            Kb  = Q / (Q + R + 1e-6)
            Kd  = 0.3 * (torch.sigmoid(self.K_net(xc).unsqueeze(-1).to(dtype)) - 0.5)
            K   = torch.clamp(Kb + Kd, 1e-4, 0.999)
            mu  = self.parallel_scan(
                torch.clamp(1.0 - K, 0.01, 0.99), K * v, self.mu_init
            ).reshape(B, T, self.E)
            yp   = self.out_proj(mu * gate)
            rg   = torch.sigmoid(self.res_gate_bias)
            out  = yp + rg * (self.res_proj(x) - yp)
            stats = {
                "_yp":    yp,             # tensor — popped before scalar averaging
                "K_mean": K.mean().item(), "K_std":  K.std().item(),
                "K_min":  K.min().item(),  "K_max":  K.max().item(),
                "A_mean": torch.clamp(1.0 - K, 0.01, 0.99).mean().item(),
                "R_mean": R.mean().item(), "res_gate_mean": rg.item(),
            }
            return out, Q, R, stats

    return KCAMambaBlock(d_model, heads, d_state)


def _make_arima_model(input_dim: int, p: int = 5, q: int = 3, hidden_dim: int = 256):
    """Neural ARIMA(p, 1, q):
    I(1)  – first-order differencing of the input series
    AR(p) – p lagged differenced values as features
    MA(q) – q lagged residuals (ε_t = dx_t − AR̂_{t-1}) as features
    MLP   – same two-hidden-layer net as the old AR baseline
    out   – undo differencing: ŷ_t = x_t + Δŷ_t
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ARIMAModel(nn.Module):
        def __init__(self, d: int, p: int, q: int, h: int):
            super().__init__()
            self.p, self.q = p, q
            # Lightweight AR used only to compute MA residuals (not in the main path)
            self.ar_quick = nn.Linear(d * p, d, bias=False)
            # Main path: AR+MA concatenated features → MLP → predicted difference
            self.net = nn.Sequential(
                nn.Linear(d * (p + q), h),
                nn.SiLU(),
                nn.Linear(h, h),
                nn.SiLU(),
                nn.Linear(h, d),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            B, T, D = x.shape
            p, q = self.p, self.q

            # I(1): first-order differencing; dx[:,0,:] = 0 (boundary)
            dx = torch.diff(x, dim=1)               # [B, T-1, D]
            dx = F.pad(dx, (0, 0, 1, 0))            # [B, T,   D]

            # AR features: windows of p lagged differenced values
            xt         = dx.transpose(1, 2)                              # [B, D, T]
            ar_windows = F.pad(xt, (p - 1, 0)).unfold(2, p, 1)[:, :, :T]  # [B,D,T,p]
            ar_feat    = ar_windows.permute(0, 2, 1, 3).reshape(B, T, D * p)

            # MA residuals: ε_t = dx_t − AR̂_{t} where AR̂ is shifted by one step
            ar_pred    = self.ar_quick(ar_feat)                          # [B, T, D]
            ar_shifted = F.pad(ar_pred[:, :-1], (0, 0, 1, 0))          # [B, T, D] (t-1)
            eps        = dx - ar_shifted                                 # [B, T, D]

            # MA features: windows of q lagged residuals
            et         = eps.transpose(1, 2)                              # [B, D, T]
            ma_windows = F.pad(et, (q - 1, 0)).unfold(2, q, 1)[:, :, :T]  # [B,D,T,q]
            ma_feat    = ma_windows.permute(0, 2, 1, 3).reshape(B, T, D * q)

            # MLP on concatenated AR+MA features → predicted Δx
            dy = self.net(torch.cat([ar_feat, ma_feat], dim=-1))        # [B, T, D]

            # Undo I(1): ŷ_t = x_t + Δŷ_t  (predicts x_{t+1})
            return x + dy                                                # [B, T, D]

    return ARIMAModel(input_dim, p, q, hidden_dim)


# ── FastAI-faithful LR range finder ────────────────────────────────────────
# 1:1 port of fastai/callback/schedule.py (LRFinder + valley suggestion).
# Source: https://github.com/fastai/fastai/blob/main/fastai/callback/schedule.py
#
# Key algorithm:
#   • Exponential LR schedule: lr = start_lr × (end_lr/start_lr)^(i/num_it)
#     start_lr=1e-7, end_lr=10, num_it=100  (fastai defaults)
#   • EMA-smoothed loss β=0.98 + bias correction  (fastai Recorder)
#   • Divergence stop:  smooth > 4 × best_smooth    (fastai LRFinder.after_batch)
#   • Trim: skip first num_it//10 and last 5 samples (fastai lr_find)
#   • Suggestion: ‘valley’ — ESRI algorithm, longest decreasing
#     subsequence (DP), take the point ≈2/3 through it  (fastai valley)
#   • Model weights restored after the probe run

def _find_lr_fastai(
    model,
    train_loader,
    device,
    start_lr: float = 1e-7,
    end_lr:   float = 10.0,
    num_it:   int   = 100,
    stop_div: bool  = True,
) -> float:
    import copy
    import torch
    import torch.nn.functional as F

    model_clone = copy.deepcopy(model).to(device)
    optimizer   = torch.optim.Adam(model_clone.parameters(), lr=start_lr)

    beta       = 0.98
    avg_loss   = 0.0
    best_smooth = float("inf")
    lrs:    list[float] = []
    losses: list[float] = []

    loader_iter = iter(train_loader)
    for i in range(num_it):
        # ─ Exponential LR schedule (fastai SchedExp) ──────────────────────────
        pos  = i / num_it
        lr_i = start_lr * (end_lr / start_lr) ** pos
        for pg in optimizer.param_groups:
            pg["lr"] = lr_i

        try:
            x_b, y_b = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x_b, y_b   = next(loader_iter)

        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad(set_to_none=True)

        res = model_clone(x_b)
        if isinstance(res, tuple) and len(res) >= 3:
            out, Q, R = res[0], res[1], res[2]
            pred = out[:, :-1];  tgt = y_b[:, :-1]
            base = F.mse_loss(pred, tgt, reduction="none").mean(dim=-1)
            var  = (R[:, :-1].mean(dim=(2, 3)) if R.dim() == 4
                    else R[:, :-1].mean(dim=-1)).clamp(1e-4, 10.0)
            loss = (0.5 * base / var + 0.5 * torch.log(var)).mean() + 0.5 * base.mean()
        else:
            out  = res[0] if isinstance(res, tuple) else res
            loss = F.mse_loss(out[:, :-1], y_b[:, :-1])

        if not torch.isfinite(loss):
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_clone.parameters(), 1.0)
        optimizer.step()

        # ─ EMA smooth loss with bias correction (fastai Recorder) ─────────────
        avg_loss = beta * avg_loss + (1.0 - beta) * loss.item()
        smooth   = avg_loss / (1.0 - beta ** (i + 1))

        if smooth < best_smooth:
            best_smooth = smooth
        elif stop_div and smooth > 4.0 * best_smooth:
            break  # diverged — fastai LRFinder.after_batch

        lrs.append(lr_i)
        losses.append(smooth)

    del model_clone

    # ─ Trim: skip first num_it//10 and last 5 (fastai lr_find) ─────────────
    trim_s = num_it // 10
    trim_e = len(lrs) - 5
    lrs_t    = lrs[trim_s:trim_e]
    losses_t = losses[trim_s:trim_e]

    if len(lrs_t) < 3:
        return 1e-3   # not enough points after trim — use fallback

    # ─ FastAI `valley` suggestion (ESRI algorithm, 1:1 port) ─────────────
    # Longest decreasing subsequence via DP; take ~2/3 point inside it.
    n = len(losses_t)
    max_start, max_end = 0, 0
    lds = [1] * n
    for i in range(1, n):
        for j in range(0, i):
            if losses_t[i] < losses_t[j] and lds[i] < lds[j] + 1:
                lds[i] = lds[j] + 1
        if lds[max_end] < lds[i]:
            max_end   = i
            max_start = max_end - lds[max_end]

    sections = (max_end - max_start) / 3
    idx      = max_start + int(sections) + int(sections / 2)
    idx      = max(0, min(idx, n - 1))

    return float(lrs_t[idx])


# ── Training loop for one model ───────────────────────────────────────────────

def _train_one(model, name: str, train_loader, test_loader, c: dict, device) -> dict:
    import torch
    import torch.nn.functional as F
    import numpy as np

    lr        = float(c["lr"])
    epochs    = int(c["epochs"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine annealing: start at lr, decay to lr/20 by final epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=lr / 20
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t_start = time.time()
    epoch_losses: list[float] = []
    model.train()

    for _ep in range(epochs):
        ep_losses: list[float] = []
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            res = model(x_b)
            if isinstance(res, tuple) and len(res) >= 3:
                out, Q, R = res[0], res[1], res[2]
                pred = out[:, :-1]; tgt = y_b[:, :-1]
                base = F.mse_loss(pred, tgt, reduction="none").mean(dim=-1)
                var  = (R[:, :-1].mean(dim=(2, 3)) if R.dim() == 4
                        else R[:, :-1].mean(dim=-1)).clamp(1e-4, 10.0)
                # Proper heteroscedastic NLL: 0.5*(base/var + log(var)) + 0.5*base
                # Equilibrium: var_opt = base (predicted uncertainty matches actual error)
                # With old 0.1 coeff, equilibrium was var=5*base → R too large → K≈1 → no filtering
                loss = (0.5 * base / var + 0.5 * torch.log(var)).mean() + 0.5 * base.mean()
            else:
                out  = res[0] if isinstance(res, tuple) else res
                loss = F.mse_loss(out[:, :-1], y_b[:, :-1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_losses.append(float(loss.item()))
        epoch_losses.append(round(float(np.mean(ep_losses)), 6))
        scheduler.step()  # CosineAnnealingLR steps once per epoch

    train_time = time.time() - t_start
    peak_vram  = (torch.cuda.max_memory_allocated() / 1024 ** 2
                  if device.type == "cuda" else 0.0)

    # ── Test ─────────────────────────────────────────────────────────────────
    model.eval()
    test_mses: list[float] = []
    kalman_acc: dict[str, list[float]] = {}
    sample_true = sample_pred_t = sample_noisy_t = None
    sample_kla_filtered: "list[float] | None" = None

    with torch.no_grad():
        for i, (x_b, y_b) in enumerate(test_loader):
            x_b, y_b = x_b.to(device), y_b.to(device)
            res = model(x_b)
            if isinstance(res, tuple) and len(res) == 4:
                out, _, _, stats = res
                yp_batch = stats.pop("_yp", None)  # tensor — remove before float() loop
                for k, v in stats.items():
                    kalman_acc.setdefault(k, []).append(float(v))
            else:
                out      = res[0] if isinstance(res, tuple) else res
                yp_batch = None
            test_mses.append(float(F.mse_loss(out[:, :-1], y_b[:, :-1]).item()))
            if i == 0:
                N = min(512, out.shape[1])
                sample_pred_t   = out[0, :N, -1].cpu()
                sample_true     = y_b[0, :N, -1].cpu()
                sample_noisy_t  = x_b[0, :N, -1].cpu()   # what the model actually received
                if yp_batch is not None:
                    sample_kla_filtered = [
                        round(float(v), 5) for v in yp_batch[0, :N, -1].cpu().numpy()
                    ]

    avg_kalman = {k: round(sum(v) / len(v), 5) for k, v in kalman_acc.items()}

    sample: dict = {}
    if sample_true is not None:
        sample = {
            "true":  [round(float(v), 5) for v in sample_true.numpy()],
            "pred":  [round(float(v), 5) for v in sample_pred_t.numpy()],
            "noisy": [round(float(v), 5) for v in sample_noisy_t.numpy()],
        }
        if sample_kla_filtered:
            sample["kla_filtered"] = sample_kla_filtered

    return {
        "name":         name,
        "params":       sum(p.numel() for p in model.parameters()),
        "epoch_losses": epoch_losses,
        "test_mse":     round(float(sum(test_mses) / max(len(test_mses), 1)), 6),
        "train_time_s": round(train_time, 2),
        "peak_vram_mb": round(peak_vram, 2),
        "kalman_stats": avg_kalman,
        "sample":       sample,
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def _run(cfg: dict) -> None:
    c = {**_DEFAULTS, **cfg}
    try:
        import torch
    except Exception as exc:
        with _lock:
            _state.update({"status": "failed", "error": str(exc), "finished_at": _now()})
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Download + preprocess dataset ────────────────────────────────────────
    try:
        with _lock:
            _state["stage"] = f"{c['dataset']} adatok letöltése..."
        train_loader, test_loader, input_dim = _load_dataset(
            c["dataset"], c["seq_len"], c["batch_size"],
            float(c.get("noise_scale", 0.0)), int(c.get("stride", 24))
        )
        with _lock:
            _state["input_dim"] = input_dim
            _state["stage"]     = "Adatok kész, modellek tréningje..."
    except Exception as exc:
        with _lock:
            _state.update({"status": "failed", "error": f"Adatbetöltés: {exc}", "finished_at": _now()})
        return

    models = [
        ("lstm",       "Vanilla LSTM",             _make_lstm(input_dim, int(c.get("lstm_hidden", 128))).to(device)),
        ("fair_mamba", "Fair Mamba",                _make_fair_mamba(input_dim, int(c.get("fair_heads", 4)), int(c.get("fair_state", 32))).to(device)),
        ("kca_mamba",  "KCA-Mamba",                 _make_kca_mamba(input_dim,  int(c.get("kca_heads",  4)), int(c.get("kca_state",  32))).to(device)),
        ("ar_model",  f"ARIMA({int(c['arima_p'])},1,{int(c['arima_q'])})",
            _make_arima_model(input_dim, int(c["arima_p"]), int(c["arima_q"]),
                              int(c.get("lstm_hidden", 128)) * 2).to(device)),
    ]

    partial: dict[str, Any] = {}
    for key, model_name, model in models:
        # ── FastAI LR finder (valley method) ────────────────────────────────
        with _lock:
            _state["stage"] = f"{model_name} — LR kereső (FastAI valley)..."
        try:
            found_lr = _find_lr_fastai(model, train_loader, device)
        except Exception:
            found_lr = 1e-3

        # ── Training with found LR + CosineAnnealing ────────────────────────
        with _lock:
            _state["stage"] = f"{model_name} tréning (LR={found_lr:.2e})..."
        try:
            result = _train_one(
                model, model_name, train_loader, test_loader,
                {**c, "lr": found_lr}, device,
            )
            result["found_lr"] = found_lr
        except Exception as exc:
            result = {"name": model_name, "error": str(exc), "test_mse": None,
                      "found_lr": found_lr}
        partial[key] = result
        with _lock:
            _state["models"] = copy.deepcopy(partial)

    with _lock:
        _state.update({
            "status":     "done",
            "finished_at": _now(),
            "config":     c,
            "device":     str(device),
            "input_dim":  input_dim,
            "models":     partial,
        })
