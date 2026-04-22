"""Ablation benchmark: KCA-Mamba variants on the 9 dashboard presets.

Variants trained side-by-side on each preset:
  1. KCA-base     — current production KCA-Mamba (Kalman-gain only)
  2. KCA+C        — adds input-dependent readout C_net (content-based read)
  3. KCA+BC       — KCA+C plus input-dependent write B_net (full Mamba selectivity
                    with Kalman gate)
  4. KCA-Dual     — two parallel Kalman scans at different timescales
                    (short-memory + long-memory), content-gated mix

All variants share Q/R/K Kalman-gain structure, parallel scan, res-gate, and
the same hyperparameters (heads=4, d_state=32) for a fair comparison.

Compared against the existing Fair-Mamba / LSTM / KCA-M / PatchTST results
from sweep_results.json + stride_sweep_results.json + patchtst_preset_results.json.

Run: python kca_variants_bench.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from adapters.offline_training.ltsf_benchmark import (  # noqa: E402
    _load_dataset, _find_lr_fastai, _train_one,
)


# ── Variants ──────────────────────────────────────────────────────────────────

def _make_kca_variant(variant: str, d_model: int, heads: int = 4, d_state: int = 32):
    """Build a KCA-Mamba variant as a single block. Returns a module that maps
    [B, T, d_model] -> (out, Q, R, stats) compatible with _train_one.
    """
    H, D, E = heads, d_state, heads * d_state

    class KCAVariant(nn.Module):
        def __init__(self):
            super().__init__()
            self.variant = variant
            self.H, self.D, self.E = H, D, E

            # Shared backbone (identical across variants)
            self.in_proj  = nn.Linear(d_model, E * 2)
            self.conv1d   = nn.Conv1d(E, E, 4, groups=E, padding=3)
            self.res_proj = nn.Linear(d_model, d_model)
            self.proj_v   = nn.Linear(E, E)
            self.Q_net    = nn.Sequential(nn.Linear(E, 64), nn.SiLU(), nn.Linear(64, H))
            self.R_net    = nn.Sequential(nn.Linear(E, 64), nn.SiLU(), nn.Linear(64, H))
            self.K_net    = nn.Sequential(nn.Linear(E, 64), nn.SiLU(), nn.Linear(64, H))
            self.q_scale  = nn.Parameter(torch.tensor(-2.0))
            self.r_scale  = nn.Parameter(torch.tensor( 1.5))
            self.mu_init  = nn.Parameter(torch.zeros(1, H, D))
            self.res_gate_bias = nn.Parameter(torch.tensor(-1.0))
            self.v_norm   = nn.LayerNorm(E)

            if variant == "base":
                self.out_proj = nn.Linear(E, d_model)

            elif variant == "C":
                # Input-dependent readout: per-token (H,D) query direction
                # Init near-identity: residual around a constant of 1
                self.C_net   = nn.Linear(E, E)
                self.out_proj = nn.Linear(E, d_model)

            elif variant == "BC":
                # Input-dependent write AND read (full Mamba-style selectivity)
                self.B_net   = nn.Linear(E, E)
                self.C_net   = nn.Linear(E, E)
                self.out_proj = nn.Linear(E, d_model)

            elif variant == "dual":
                # Second Kalman scan with long-memory init:
                #   q_scale_long=-4 → softplus≈0.018, K_long≈0.01, A_long≈0.99, memory ~100 steps
                # The first (short) scan uses the standard q_scale/r_scale (~14 steps).
                self.q_scale_long = nn.Parameter(torch.tensor(-4.0))
                self.r_scale_long = nn.Parameter(torch.tensor( 2.5))
                self.mu_init_long = nn.Parameter(torch.zeros(1, H, D))
                # Content-based gate: softmax over (short, long)
                self.scale_gate   = nn.Linear(E, 2)
                # Readout from concatenated short+long state
                self.out_proj     = nn.Linear(E * 2, d_model)

            else:
                raise ValueError(f"Unknown KCA variant: {variant!r}")

        def parallel_scan(self, A, B_scan, mu0):
            T, step = A.shape[1], 1
            while step < T:
                Ar = A[:, step:]
                As = A[:, :-step]
                Bs = B_scan[:, :-step]
                A = torch.cat([A[:, :step], Ar * As], dim=1)
                B_scan = torch.cat([B_scan[:, :step], B_scan[:, step:] + Ar * Bs], dim=1)
                step *= 2
            return B_scan + A * mu0.unsqueeze(1)

        def _kalman_scan(self, xc, v, q_scale, r_scale, mu_init_param, B_override=None):
            """Run one Kalman-gain scan. Returns mu_flat [B, T, E]."""
            Q = (F.softplus(self.Q_net(xc)) * F.softplus(q_scale)).unsqueeze(-1)
            R = (F.softplus(self.R_net(xc)) * F.softplus(r_scale)).unsqueeze(-1)
            Kb = Q / (Q + R + 1e-6)
            Kd = 0.3 * (torch.sigmoid(self.K_net(xc).unsqueeze(-1).to(xc.dtype)) - 0.5)
            K  = torch.clamp(Kb + Kd, 1e-4, 0.999)
            write = v if B_override is None else B_override
            mu = self.parallel_scan(
                torch.clamp(1.0 - K, 0.01, 0.99),
                K * write,
                mu_init_param,
            )
            return mu.reshape(mu.shape[0], mu.shape[1], self.E), Q, R, K

        def forward(self, x):
            B_, T, _ = x.shape
            xw, xg = self.in_proj(x).chunk(2, dim=-1)
            gate   = F.silu(xg)
            xc     = F.silu(self.conv1d(xw.transpose(1, 2))[:, :, :T].transpose(1, 2))
            v      = self.v_norm(self.proj_v(xc)).view(B_, T, self.H, self.D)

            if self.variant == "base":
                mu_flat, Q, R, K = self._kalman_scan(xc, v, self.q_scale, self.r_scale, self.mu_init)
                yp = self.out_proj(mu_flat * gate)

            elif self.variant == "C":
                # Input-dependent readout multiplier. Init C near 1.0 so base
                # behavior is preserved at step 0.
                C = 1.0 + 0.1 * self.C_net(xc)                     # [B, T, E]
                mu_flat, Q, R, K = self._kalman_scan(xc, v, self.q_scale, self.r_scale, self.mu_init)
                yp = self.out_proj(mu_flat * gate * C)

            elif self.variant == "BC":
                # Write-direction B is input-dependent (on top of K*v weighting),
                # read-direction C is input-dependent.
                B_write = (1.0 + 0.1 * self.B_net(xc)).view(B_, T, self.H, self.D)
                C       = 1.0 + 0.1 * self.C_net(xc)
                mu_flat, Q, R, K = self._kalman_scan(
                    xc, v, self.q_scale, self.r_scale, self.mu_init,
                    B_override=v * B_write,
                )
                yp = self.out_proj(mu_flat * gate * C)

            elif self.variant == "dual":
                mu_short, Q, R, K = self._kalman_scan(
                    xc, v, self.q_scale, self.r_scale, self.mu_init
                )
                mu_long, _, _, _ = self._kalman_scan(
                    xc, v, self.q_scale_long, self.r_scale_long, self.mu_init_long
                )
                # Content-gated mix, then concat for the out_proj (richer readout)
                gmix = F.softmax(self.scale_gate(xc), dim=-1)        # [B, T, 2]
                g_s, g_l = gmix[..., 0:1], gmix[..., 1:2]
                short_w = mu_short * g_s.expand(-1, -1, self.E)
                long_w  = mu_long  * g_l.expand(-1, -1, self.E)
                cat = torch.cat([short_w, long_w], dim=-1) * torch.cat([gate, gate], dim=-1)
                yp = self.out_proj(cat)

            rg = torch.sigmoid(self.res_gate_bias)
            out = yp + rg * (self.res_proj(x) - yp)
            stats = {
                "K_mean": float(K.mean().item()),
                "K_std":  float(K.std().item()),
                "A_mean": float(torch.clamp(1.0 - K, 0.01, 0.99).mean().item()),
                "R_mean": float(R.mean().item()),
                "res_gate_mean": float(rg.item()),
            }
            return out, Q, R, stats

    return KCAVariant()


# ── Preset + driver (mirrors patchtst_preset_bench.py) ────────────────────────

PRESETS = [
    ("Exchange", 192, 1,  0.0, "sweep"),
    ("Exchange", 336, 2,  3.0, "sweep"),
    ("Exchange", 336, 2,  1.0, "sweep"),
    ("Exchange", 512, 2,  3.0, "sweep"),
    ("Exchange", 192, 1,  5.0, "stride_sweep"),
    ("Exchange", 512, 1,  5.0, "stride_sweep"),
    ("ETTm1",     96, 4,  0.0, "sweep"),
    ("ETTm1",    512, 16, 5.0, "sweep"),
    ("Weather",  336, 16, 5.0, "sweep"),
]

VARIANTS = ["base", "C", "BC", "dual"]
EPOCHS     = 10
BATCH_SIZE = 64
ROOT       = pathlib.Path(__file__).parent
OUT_FILE   = ROOT / "kca_variants_results.json"


def _load_existing_sweeps() -> dict:
    """test_mse lookup: (ds, seq, noise, model) -> float from saved runs."""
    out: dict = {}
    sweep = json.loads((ROOT / "sweep_results.json").read_text())
    for _, v in sweep.items():
        if isinstance(v, dict) and "test_mse" in v:
            out[(v["dataset"], v["seq_len"], float(v["noise"]), v["model"])] = v["test_mse"]
    stride_sweep = json.loads((ROOT / "stride_sweep_results.json").read_text())
    for k, v in stride_sweep.items():
        if not isinstance(v, dict) or "test_mse" not in v:
            continue
        parts = k.split("|")
        if len(parts) == 5 and parts[2] == "s1":
            ds = parts[0]
            seq = int(parts[1].removeprefix("seq"))
            noise = float(parts[3])
            mdl = parts[4]
            out.setdefault((ds, seq, noise, mdl), v["test_mse"])
    # PatchTST results (saved by patchtst_preset_bench.py)
    ptst_file = ROOT / "patchtst_preset_results.json"
    if ptst_file.exists():
        ptst = json.loads(ptst_file.read_text())
        for k, v in ptst.items():
            if isinstance(v, dict) and "test_mse" in v:
                parts = k.split("|")
                ds = parts[0]
                seq = int(parts[1])
                noise = float(parts[3])
                out[(ds, seq, noise, "patchtst")] = v["test_mse"]
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    sweeps = _load_existing_sweeps()
    results: dict = {}
    if OUT_FILE.exists():
        try:
            results = json.loads(OUT_FILE.read_text())
        except Exception:
            results = {}

    t_start = time.time()
    total_cells = len(PRESETS) * len(VARIANTS)
    cell = 0
    for (ds, seq, stride, noise, _src) in PRESETS:
        # Load the dataset once per preset (not per variant)
        try:
            train_loader, test_loader, input_dim = _load_dataset(
                ds, seq, BATCH_SIZE, noise_scale=noise, stride=stride
            )
        except Exception as exc:
            print(f"[SKIP] {ds} seq={seq} σ={noise}: {exc}")
            continue

        for variant in VARIANTS:
            cell += 1
            key = f"{ds}|{seq}|s{stride}|{noise:.1f}|kca_{variant}"
            if key in results and "test_mse" in results[key]:
                print(f"[{cell:2d}/{total_cells}] {key}  (cached, skip)")
                continue

            print(f"[{cell:2d}/{total_cells}] {ds} seq={seq} σ={noise}  →  KCA-{variant}")
            model = _make_kca_variant(variant, input_dim, heads=4, d_state=32).to(device)
            n_params = sum(p.numel() for p in model.parameters())

            try:
                found_lr = _find_lr_fastai(model, train_loader, device)
                found_lr = max(1e-5, min(found_lr, 5e-3))
            except Exception:
                found_lr = 1e-3

            try:
                res = _train_one(
                    model, f"kca_{variant}", train_loader, test_loader,
                    {"lr": found_lr, "epochs": EPOCHS}, device,
                )
            except Exception as exc:
                print(f"  [ERR] train: {exc}")
                results[key] = {"error": f"train: {exc}"}
                OUT_FILE.write_text(json.dumps(results, indent=2))
                continue

            results[key] = {
                "dataset": ds, "seq_len": seq, "stride": stride, "noise": noise,
                "variant": variant, "n_params": n_params,
                "found_lr": round(found_lr, 8),
                "test_mse": res["test_mse"],
                "train_time_s": res["train_time_s"],
                "peak_vram_mb": res.get("peak_vram_mb"),
            }
            OUT_FILE.write_text(json.dumps(results, indent=2))
            print(
                f"  ✓ mse={res['test_mse']:.4f}  params={n_params:>6d}  "
                f"lr={found_lr:.2e}  t={res['train_time_s']:.1f}s"
            )

    total_min = (time.time() - t_start) / 60
    print(f"\nDone in {total_min:.1f} min.\n")

    # ── Build comparison table ────────────────────────────────────────────────
    print("=" * 120)
    hdr = (
        f"{'preset':<24} "
        f"{'Fair-M':>8} {'KCA-M':>8} {'PatchTST':>9} "
        f"{'KCA+C':>8} {'KCA+BC':>8} {'KCA-Dual':>9} "
        f"{'Δ best vs base':>16}"
    )
    print(hdr)
    print("-" * 120)

    for (ds, seq, stride, noise, _src) in PRESETS:
        tag = f"{ds[:3]}_{seq}_s{stride}_σ{noise:.0f}"
        fair = sweeps.get((ds, seq, noise, "fair_mamba"))
        kca  = sweeps.get((ds, seq, noise, "kca_mamba"))
        ptst = sweeps.get((ds, seq, noise, "patchtst"))

        def _mse(v):
            k = f"{ds}|{seq}|s{stride}|{noise:.1f}|kca_{v}"
            r = results.get(k, {})
            return r.get("test_mse")

        mse_c    = _mse("C")
        mse_bc   = _mse("BC")
        mse_dual = _mse("dual")

        def fmt(v): return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

        # Best among C/BC/dual vs KCA-M base
        best = None
        for name, v in [("+C", mse_c), ("+BC", mse_bc), ("-Dual", mse_dual)]:
            if isinstance(v, (int, float)):
                if best is None or v < best[1]:
                    best = (name, v)

        if best is not None and isinstance(kca, (int, float)):
            pct = (best[1] - kca) / kca * 100
            delta_str = f"{best[0]}: {pct:+6.1f}%"
        else:
            delta_str = "—"

        print(
            f"{tag:<24} "
            f"{fmt(fair):>8} {fmt(kca):>8} {fmt(ptst):>9} "
            f"{fmt(mse_c):>8} {fmt(mse_bc):>8} {fmt(mse_dual):>9} "
            f"{delta_str:>16}"
        )
    print("=" * 120)
    print("\nΔ < 0 : a variáns JOBB mint a KCA-base. Δ > 0 : rosszabb.")
    print(f"\nResults saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
