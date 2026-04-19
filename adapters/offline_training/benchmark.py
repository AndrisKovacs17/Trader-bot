"""Synthetic benchmark: KLA-Mamba vs Plain Mamba on a latent-tracking task.

Runs both models on an auto-regressive latent-state tracking problem with:
  - Regime-switching AR(1) latent process
  - Noisy / bursty / blackout observations (heavy-tail sensor model)
Returns structured JSON-serialisable results consumable by the dashboard.
"""
from __future__ import annotations

import copy
import threading
import time
from datetime import datetime, timezone
from typing import Any

# ── Shared state ─────────────────────────────────────────────────────────────
_lock: threading.Lock = threading.Lock()
_state: dict[str, Any] = {"status": "idle"}

# ── Default config ────────────────────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "batch": 32,
    "seq_len": 1024,
    "input_dim": 128,
    "heads": 4,
    "state_dim": 32,      # E = heads × state_dim = 128
    "train_steps": 50,
    "warmup": 10,
    "train_batches": 32,
    "test_batches": 32,
    "seed_train": 42,
    "seed_test": 4242,
    "lr": 5e-4,
    "sample_points": 256, # time-steps to return for the plot
}


def get_state() -> dict:
    with _lock:
        return copy.deepcopy(_state)


def trigger(cfg: dict | None = None) -> bool:
    """Start the benchmark in a background thread.
    Returns False if a run is already in progress."""
    with _lock:
        if _state.get("status") == "running":
            return False
        _state.clear()
        _state["status"] = "running"
        _state["started_at"] = _now()
    t = threading.Thread(target=_run, kwargs={"cfg": cfg or {}}, daemon=True)
    t.start()
    return True


# ── Utilities ─────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_list(t: "torch.Tensor") -> list[float]:  # type: ignore[name-defined]
    import math
    return [v if math.isfinite(v) else 0.0 for v in t.detach().cpu().float().numpy().tolist()]


# ── Data generation (matching the user's notebook) ────────────────────────────

def _generate_sequence(batch: int, T: int, dim: int, device: "torch.device"):  # type: ignore[name-defined]
    import torch
    latent = torch.zeros(batch, T, dim, device=device)
    obs    = torch.zeros(batch, T, dim, device=device)

    regime   = torch.randint(0, 2, (batch,), device=device)
    broken   = torch.zeros(batch, 1, device=device)
    blackout = torch.zeros(batch, 1, device=device)

    for t in range(1, T):
        # 1) Latent process
        flip   = torch.rand(batch, device=device) < 0.003
        regime = torch.where(flip, 1 - regime, regime)
        coef   = torch.where(regime == 0, torch.tensor(1.15, device=device),
                             torch.tensor(-0.75, device=device)).unsqueeze(-1)
        latent[:, t] = (
            torch.tanh(coef * latent[:, t - 1] + 0.1 * torch.sin(0.3 * t + latent[:, t - 1]))
            + 0.01 * torch.randn(batch, dim, device=device)
        )
        latent[:, t] += 0.001 * t
        shock = (torch.rand(batch, device=device) < 0.01).unsqueeze(-1)
        latent[:, t] += shock * (0.5 * torch.randn(batch, dim, device=device))
        drift = (torch.rand(batch, device=device) < 0.03).unsqueeze(-1)
        latent[:, t] += drift * 0.2

        # 2) Observation model
        breakdown = torch.rand(batch, 1, device=device) < 0.01
        fix       = torch.rand(batch, 1, device=device) < 0.20
        broken = torch.where(broken > 0.5, (1.0 - fix.float()), breakdown.float())

        start_bo  = torch.rand(batch, 1, device=device) < 0.02
        keep_bo   = (blackout > 0.5) & (torch.rand(batch, 1, device=device) < 0.80)
        blackout  = torch.where(blackout > 0.5, keep_bo.float(), start_bo.float())

        state_scale  = 0.05 + 0.2 * latent[:, t].abs().mean(-1, keepdim=True)
        broken_scale = torch.where(broken > 0.5, torch.tensor(3.0, device=device),
                                   torch.tensor(1.0, device=device))
        burst = (torch.rand(batch, 1, device=device) < 0.02).float()
        noise_scale  = state_scale * broken_scale * (1.0 + 4.0 * burst)

        eps  = torch.randn(batch, dim, device=device)
        heavy = eps + 0.05 * (eps ** 3)
        obs_t = latent[:, t] + noise_scale * heavy
        obs[:, t] = torch.where(blackout > 0.5, torch.zeros_like(obs_t), obs_t)

    return obs, latent


def _make_dataset(n: int, seed: int, c: dict, device: "torch.device"):  # type: ignore[name-defined]
    import torch
    obs_list, lat_list = [], []
    for i in range(n):
        torch.manual_seed(seed + i)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + i)
        obs, lat = _generate_sequence(c["batch"], c["seq_len"], c["input_dim"], device)
        obs_list.append(obs)
        lat_list.append(lat)
    return obs_list, lat_list


# ── Plain Mamba block (fair baseline, matches user's FairMambaBlock) ──────────

def _make_plain_mamba(d_model: int, heads: int, d_state: int) -> "torch.nn.Module":  # type: ignore[name-defined]
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FairMambaBlock(nn.Module):
        def __init__(self, d_model: int, heads: int, d_state: int, kernel_size: int = 4):
            super().__init__()
            self.H = heads
            self.D = d_state
            self.E = heads * d_state

            self.in_proj  = nn.Linear(d_model, self.E * 2)
            self.conv1d   = nn.Conv1d(self.E, self.E, kernel_size, groups=self.E,
                                      padding=kernel_size - 1)
            self.dt_net   = nn.Sequential(
                nn.Linear(self.E, 64), nn.SiLU(), nn.Linear(64, self.E)
            )
            self.proj_B   = nn.Linear(self.E, self.E)
            self.proj_C   = nn.Linear(self.E, self.E)
            self.out_proj = nn.Linear(self.E, d_model)

        def parallel_scan(self, A: torch.Tensor, B: torch.Tensor, h_init: torch.Tensor) -> torch.Tensor:
            T, step = A.shape[1], 1
            while step < T:
                A_right   = A[:, step:]
                A_shifted = A[:, :-step]
                B_shifted = B[:, :-step]
                A_new = A_right * A_shifted
                B_new = B[:, step:] + A_right * B_shifted
                A = torch.cat([A[:, :step], A_new], dim=1)
                B = torch.cat([B[:, :step], B_new], dim=1)
                step *= 2
            return B + A * h_init.unsqueeze(1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B_sz, T, _ = x.shape
            x_wide, x_gate = self.in_proj(x).chunk(2, dim=-1)
            gate = F.silu(x_gate)
            x_core = self.conv1d(x_wide.transpose(1, 2))[:, :, :T].transpose(1, 2)
            x_core = F.silu(x_core)

            dt_seq = F.softplus(self.dt_net(x_core)).clamp(1e-4, 0.1)
            A_scan = torch.exp(-dt_seq)
            B_scan = dt_seq * self.proj_B(x_core)
            h_init = torch.zeros(B_sz, self.E, device=x.device)
            h_all  = self.parallel_scan(A_scan, B_scan, h_init)
            y_ssm  = self.proj_C(x_core) * h_all
            return x + self.out_proj(y_ssm * gate)

    return FairMambaBlock(d_model, heads, d_state)


# ── Training loop for one model ───────────────────────────────────────────────

def _run_one(
    model: "torch.nn.Module",  # type: ignore[name-defined]
    name: str,
    train_obs: list,
    train_lat: list,
    test_obs: list,
    test_lat: list,
    c: dict,
    device: "torch.device",  # type: ignore[name-defined]
) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    is_kla = name != "Plain Mamba"
    optimizer = torch.optim.Adam(model.parameters(), lr=float(c["lr"]))
    loss_fn   = nn.MSELoss(reduction="none")

    # ── Warmup ────────────────────────────────────────────────────────────────
    for _ in range(c["warmup"]):
        out_raw = model(train_obs[0])
        out = out_raw[0] if isinstance(out_raw, tuple) else out_raw
        nn.MSELoss()(out[:, :-1], train_lat[0][:, 1:]).backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ── Training ──────────────────────────────────────────────────────────────
    train_losses: list[float] = []
    t_start = time.time()

    for step in range(int(c["train_steps"])):
        idx = step % len(train_obs)
        obs_b, lat_b = train_obs[idx], train_lat[idx]
        optimizer.zero_grad(set_to_none=True)

        out_raw = model(obs_b)
        if isinstance(out_raw, tuple):
            out, Q_seq, R_seq = out_raw[0], out_raw[1], out_raw[2]
        else:
            out, Q_seq, R_seq = out_raw, None, None

        out  = torch.nan_to_num(out)
        pred = out[:, :-1].float()
        tgt  = lat_b[:, 1:].float()
        base_loss = loss_fn(pred, tgt).mean(dim=-1)  # [B, T-1]

        if R_seq is not None:
            variance = R_seq[:, :-1].mean(dim=(2, 3)).clamp(1e-4, 10.0)
            mse_part = base_loss / (2.0 * variance)
            penalty  = 0.1 * torch.log(variance)
            loss = (mse_part + penalty).mean() + 0.5 * base_loss.mean()
        else:
            loss = base_loss.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(float(loss.item()))

    elapsed = time.time() - t_start
    tokens = int(c["batch"]) * int(c["seq_len"]) * int(c["train_steps"])
    throughput = int(tokens / max(elapsed, 1e-9))
    peak_vram  = (torch.cuda.max_memory_allocated() / 1024 ** 2
                  if device.type == "cuda" else 0.0)

    # ── Testing ───────────────────────────────────────────────────────────────
    model.eval()
    test_mse_vals: list[float] = []
    kalman_stats: dict[str, list[float]] = {
        k: [] for k in ("K_mean", "K_std", "K_min", "K_max", "A_mean", "R_mean", "res_gate_mean")
    }

    with torch.no_grad():
        for obs_b, lat_b in zip(test_obs, test_lat):
            out_raw = model(obs_b)
            if isinstance(out_raw, tuple) and len(out_raw) == 4:
                out, _, _, stats = out_raw
                for k in kalman_stats:
                    if k in stats:
                        kalman_stats[k].append(float(stats[k]))
            else:
                out = out_raw[0] if isinstance(out_raw, tuple) else out_raw
            pred = out[:, :-1].float()
            tgt  = lat_b[:, 1:].float()
            test_mse_vals.append(float(F.mse_loss(pred, tgt).item()))

    test_loss = sum(test_mse_vals) / max(len(test_mse_vals), 1)
    avg_kalman = {k: (sum(v) / len(v)) for k, v in kalman_stats.items() if v}

    # ── Sample prediction for the plot ───────────────────────────────────────
    def _sample(obs_b: "torch.Tensor", lat_b: "torch.Tensor") -> dict:
        with torch.no_grad():
            out_raw = model(obs_b)
            pred_s  = (out_raw[0] if isinstance(out_raw, tuple) else out_raw)[0, :, 0]
        N = min(int(c["sample_points"]), obs_b.shape[1])
        return {
            "obs":    _to_list(obs_b[0, :N, 0]),
            "latent": _to_list(lat_b[0, :N, 0]),
            "pred":   _to_list(pred_s[:N]),
        }

    sample_train = _sample(train_obs[0], train_lat[0])
    sample_test  = _sample(test_obs[0],  test_lat[0])

    return {
        "name": name,
        "params": sum(p.numel() for p in model.parameters()),
        "train_losses": [round(v, 6) for v in train_losses],
        "test_loss": round(test_loss, 6),
        "peak_vram_mb": round(peak_vram, 2),
        "throughput": throughput,
        "kalman_stats": {k: round(v, 5) for k, v in avg_kalman.items()},
        "sample_train": sample_train,
        "sample_test":  sample_test,
    }


# ── Standalone KLA-Mamba block (1:1 match with notebook) ─────────────────────

def _make_kla_mamba(d_model: int, heads: int, d_state: int) -> "torch.nn.Module":  # type: ignore[name-defined]
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class KLAMambaBlock(nn.Module):
        def __init__(self, d_model: int, heads: int, d_state: int, kernel_size: int = 4):
            super().__init__()
            self.H, self.D, self.E = heads, d_state, heads * d_state

            self.q_scale = nn.Parameter(torch.tensor(-2.5))
            self.r_scale = nn.Parameter(torch.tensor(1.5))

            self.in_proj  = nn.Linear(d_model, self.E * 2)
            self.conv1d   = nn.Conv1d(self.E, self.E, kernel_size, groups=self.E, padding=kernel_size - 1)
            self.out_proj = nn.Linear(self.E, d_model)
            self.res_proj = nn.Linear(d_model, d_model)
            self.proj_v   = nn.Linear(self.E, self.E)

            self.Q_net = nn.Sequential(nn.Linear(self.E, 64), nn.SiLU(), nn.Linear(64, self.H))
            self.R_net = nn.Sequential(nn.Linear(self.E, 64), nn.SiLU(), nn.Linear(64, self.H))
            self.K_net = nn.Sequential(nn.Linear(self.E, 64), nn.SiLU(), nn.Linear(64, self.H))

            self.mu_init      = nn.Parameter(torch.zeros(1, self.H, self.D))
            self.res_gate_bias = nn.Parameter(torch.tensor(-1.0))
            self.v_norm       = nn.LayerNorm(self.E)

        def parallel_scan(self, A: torch.Tensor, B: torch.Tensor, mu_init: torch.Tensor) -> torch.Tensor:
            T, step = A.shape[1], 1
            while step < T:
                A_shifted, B_shifted = A[:, :-step], B[:, :-step]
                A = torch.cat([A[:, :step], A[:, step:] * A_shifted], dim=1)
                B = torch.cat([B[:, :step], B[:, step:] + A[:, step:] * B_shifted], dim=1)
                step *= 2
            return B + A * mu_init.unsqueeze(1)

        def forward(self, x: torch.Tensor):
            B_sz, T, _ = x.shape
            dtype = x.dtype

            x_cw, x_gate = self.in_proj(x).chunk(2, dim=-1)
            gate = F.silu(x_gate)

            x_core = self.conv1d(x_cw.transpose(1, 2))[:, :, :T].transpose(1, 2)
            x_core = F.silu(x_core)

            v_seq = self.v_norm(self.proj_v(x_core)).view(B_sz, T, self.H, self.D)

            Q = (F.softplus(self.Q_net(x_core)) * F.softplus(self.q_scale)).unsqueeze(-1)
            R = (F.softplus(self.R_net(x_core)) * F.softplus(self.r_scale)).unsqueeze(-1)

            K_base = Q / (Q + R + 1e-6)
            K_corr = 0.5 * (torch.tanh(self.K_net(x_core)).unsqueeze(-1).to(dtype) + 1.0)
            K_seq  = torch.clamp(0.7 * K_base + 0.3 * K_corr, 1e-4, 0.999)

            A_scan = torch.clamp(1.0 - K_seq, 0.05, 0.995)
            B_scan = K_seq * v_seq

            mu_all = self.parallel_scan(A_scan, B_scan, self.mu_init).reshape(B_sz, T, self.E)
            y_p = self.out_proj(mu_all * gate)

            res_g = torch.sigmoid(self.res_gate_bias)
            out   = y_p + res_g * (self.res_proj(x) - y_p)

            debug_stats = {
                "K_mean":       K_seq.mean().item(),
                "K_std":        K_seq.std().item(),
                "K_min":        K_seq.min().item(),
                "K_max":        K_seq.max().item(),
                "A_mean":       A_scan.mean().item(),
                "R_mean":       R.mean().item(),
                "res_gate_mean": res_g.item(),
            }
            return out, Q, R, debug_stats

    return KLAMambaBlock(d_model, heads, d_state)


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

    # Shared datasets (same random sequences for both models)
    train_obs, train_lat = _make_dataset(c["train_batches"], c["seed_train"], c, device)
    test_obs,  test_lat  = _make_dataset(c["test_batches"],  c["seed_test"],  c, device)

    models_to_run = [
        ("plain", "Plain Mamba",
         _make_plain_mamba(c["input_dim"], c["heads"], c["state_dim"]).to(device)),
        ("kla",   "KLA-Mamba",
         _make_kla_mamba(c["input_dim"], c["heads"], c["state_dim"]).to(device)),
    ]

    partial: dict[str, Any] = {}
    for model_key, model_name, model in models_to_run:
        try:
            result = _run_one(model, model_name,
                              train_obs, train_lat, test_obs, test_lat, c, device)
        except Exception as exc:
            result = {"name": model_name, "error": str(exc)}
        partial[model_key] = result
        # Publish partial progress so the UI can show one model at a time
        with _lock:
            _state["models"] = copy.deepcopy(partial)

    with _lock:
        _state.update({
            "status": "done",
            "finished_at": _now(),
            "config": c,
            "device": str(device),
            "models": partial,
        })
