"""
TDK full sweep — Fair Mamba vs KCMamba (+ ablations).

Models:
  fair_mamba   — baseline (no Kalman gate)
  kca_mamba    — full KCMamba (adaptive Q/R/K)
  kca_fixed_k  — ablation (b): K ≡ 0.5 constant
  kca_kbase    — ablation (c): K = Q/(Q+R) only, no K_net

Main grid:
  Datasets : Exchange, ETTh1, ETTh2, ETTm1, SynthLDS, Lorenz
  T        : 96, 192, 512, 1024
  s        : 1, 4, 8, 16, 32, 64
  noise    : 0.0, 1.0, 3.0, 5.0
  seeds    : 3  (mean ± std reported)

Ablation grid (Exchange only, T=96, s in {1,16}):
  Same noise x seeds, adds kca_fixed_k and kca_kbase.

Results saved incrementally to tdk_results.json after every cell.

Usage:
  python tdk_sweep.py                          # full sweep
  python tdk_sweep.py --resume                 # continue interrupted run
  python tdk_sweep.py --fast                   # quick smoke-test (~8 min)
  python tdk_sweep.py --datasets Exchange ETTh1 --seq-lens 96 192
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from statistics import mean, stdev

# ── Sweep defaults ─────────────────────────────────────────────────────────────
DATASETS    = ["Exchange", "ETTh1", "ETTh2", "ETTm1", "SynthLDS", "Lorenz"]
SEQ_LENS    = [96, 192, 512, 1024]
STRIDES     = [1, 4, 8, 16, 32, 64]
NOISES      = [0.0, 1.0, 3.0, 5.0]
N_SEEDS     = 3
EPOCHS      = 10
BATCH_SIZE  = 64
LR_WARMUP   = True
MIN_WINDOWS = 10

OUT_FILE    = pathlib.Path(__file__).parent / "tdk_results.json"

MODELS_MAIN     = ["fair_mamba", "kca_mamba", "dlinear"]
MODELS_ABLATION = ["kca_fixed_k", "kca_kbase"]

MODEL_CFGS: dict[str, dict] = {m: {"heads": 4, "state": 32}
                                for m in MODELS_MAIN + MODELS_ABLATION}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _cell_key(ds: str, seq: int, stride: int, noise: float, model: str) -> str:
    return f"{ds}|T{seq}|s{stride}|s{noise:.1f}|{model}"


def _load_results(path: pathlib.Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def _save(path: pathlib.Path, results: dict) -> None:
    path.write_text(json.dumps(results, indent=2))


def _progress(done: int, total: int, start_t: float, label: str) -> None:
    elapsed = time.time() - start_t
    eta = (elapsed / done * (total - done)) if done > 0 else 0.0
    bar_w  = 28
    filled = int(bar_w * done / max(total, 1))
    bar    = "#" * filled + "." * (bar_w - filled)
    print(
        f"\r[{bar}] {done}/{total}  "
        f"elapsed {elapsed / 60:.1f}m  ETA {eta / 60:.1f}m  | {label}",
        end="", flush=True,
    )


def _build_cells(
    datasets: list[str],
    seq_lens: list[int],
    strides:  list[int],
    noises:   list[float],
    models:   list[str],
) -> list[tuple]:
    cells = []
    for ds in datasets:
        for seq in seq_lens:
            for stride in strides:
                for noise in noises:
                    for mdl in models:
                        # Ablation models only on Exchange / T=96 / s in {1,16}
                        if mdl in MODELS_ABLATION:
                            if ds != "Exchange" or seq != 96 or stride not in (1, 16):
                                continue
                        cells.append((ds, seq, stride, noise, mdl))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TDK full sweep -- Fair Mamba vs KCMamba"
    )
    parser.add_argument("--resume",       action="store_true",
                        help="Skip cells already present in tdk_results.json")
    parser.add_argument("--fast",         action="store_true",
                        help="Smoke-test: Exchange T=96 s={1,16} sigma={0,3} 1 seed 3 epochs")
    parser.add_argument("--no-ablation",  action="store_true",
                        help="Skip ablation models (kca_fixed_k, kca_kbase)")
    parser.add_argument("--datasets",     nargs="+", default=DATASETS)
    parser.add_argument("--seq-lens",     nargs="+", type=int,   default=SEQ_LENS)
    parser.add_argument("--strides",      nargs="+", type=int,   default=STRIDES)
    parser.add_argument("--noises",       nargs="+", type=float, default=NOISES)
    parser.add_argument("--models",       nargs="+",
                        default=MODELS_MAIN + MODELS_ABLATION)
    parser.add_argument("--seeds",        type=int, default=N_SEEDS)
    parser.add_argument("--epochs",       type=int, default=EPOCHS)
    args = parser.parse_args()

    if args.fast:
        args.datasets  = ["Exchange"]
        args.seq_lens  = [96]
        args.strides   = [1, 16]
        args.noises    = [0.0, 3.0]
        args.models    = MODELS_MAIN
        args.seeds     = 1
        args.epochs    = 3

    if args.no_ablation:
        args.models = [m for m in args.models if m not in MODELS_ABLATION]

    try:
        import torch
        import numpy as np  # noqa: F401
    except ImportError as exc:
        sys.exit(f"Missing dependency: {exc}")

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    try:
        from adapters.offline_training.ltsf_benchmark import (
            _make_fair_mamba,
            _make_kca_mamba,
            _make_kca_fixed_k,
            _make_kca_kbase,
            _make_dlinear,
            _load_dataset,
            _find_lr_fastai,
            _train_one,
        )
    except ImportError as exc:
        sys.exit(f"Cannot import benchmark helpers: {exc}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    def make_model(name: str, d: int, seq: int = 96):
        cfg = MODEL_CFGS.get(name, {"heads": 4, "state": 32})
        h, s = cfg["heads"], cfg["state"]
        if name == "fair_mamba":  return _make_fair_mamba(d, h, s).to(device)
        if name == "kca_mamba":   return _make_kca_mamba(d, h, s).to(device)
        if name == "kca_fixed_k": return _make_kca_fixed_k(d, h, s).to(device)
        if name == "kca_kbase":   return _make_kca_kbase(d, h, s).to(device)
        if name == "dlinear":     return _make_dlinear(d, seq_len=seq).to(device)
        raise ValueError(f"Unknown model: {name}")

    results   = _load_results(OUT_FILE) if args.resume else {}
    all_cells = _build_cells(args.datasets, args.seq_lens, args.strides,
                             args.noises, args.models)
    todo = [
        (ds, seq, stride, noise, mdl, _cell_key(ds, seq, stride, noise, mdl))
        for ds, seq, stride, noise, mdl in all_cells
        if not (args.resume and _cell_key(ds, seq, stride, noise, mdl) in results)
    ]

    total = len(todo)
    if total == 0:
        print("Nothing to run. Remove tdk_results.json or omit --resume to rerun.")
        return

    print(f"\nTotal cells to run : {total}")
    print(f"Seeds per cell     : {args.seeds}")
    print(f"Epochs per seed    : {args.epochs}")
    print(f"Output             : {OUT_FILE}\n")

    sweep_start = time.time()

    for cell_idx, (ds, seq, stride, noise, mdl, key) in enumerate(todo):
        label = f"{ds} T={seq} s={stride} sigma={noise:.1f} {mdl}"
        _progress(cell_idx, total, sweep_start, label)

        try:
            train_loader, test_loader, input_dim = _load_dataset(
                ds, seq, BATCH_SIZE, noise_scale=noise, stride=stride
            )
        except Exception as exc:
            print(f"\n  [SKIP-LOAD] {label}: {exc}")
            results[key] = {"error": f"load:{exc}", "label": label}
            _save(OUT_FILE, results)
            continue

        n_windows = len(train_loader.dataset)
        if n_windows < MIN_WINDOWS:
            results[key] = {"skipped": f"only {n_windows} train windows", "label": label}
            _save(OUT_FILE, results)
            continue

        # LR finder once (probe run; internal deep-copy, original model unchanged)
        if LR_WARMUP and len(train_loader) >= 5:
            try:
                probe    = make_model(mdl, input_dim, seq)
                found_lr = _find_lr_fastai(probe, train_loader, device)
                found_lr = max(1e-5, min(found_lr, 5e-3))
                del probe
            except Exception:
                found_lr = 1e-3
        else:
            found_lr = 1e-3

        seed_mses:  list[float] = []
        seed_maes:  list[float] = []
        seed_ks:    list[dict]  = []
        seed_times: list[float] = []
        params: int | None = None

        for si in range(args.seeds):
            seed_val = 42 + si * 1337
            torch.manual_seed(seed_val)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed_val)
            try:
                import numpy as _np
                _np.random.seed(seed_val % (2 ** 31))
            except Exception:
                pass

            m = make_model(mdl, input_dim, seq)
            params = sum(p.numel() for p in m.parameters())

            try:
                r = _train_one(
                    m, mdl, train_loader, test_loader,
                    {"lr": found_lr, "epochs": args.epochs},
                    device,
                )
                seed_mses.append(r["test_mse"])
                seed_maes.append(r.get("test_mae", float("nan")))
                if r.get("kalman_stats"):
                    seed_ks.append(r["kalman_stats"])
                seed_times.append(r.get("train_time_s") or 0.0)
            except Exception as exc:
                print(f"\n  [ERR seed {si}] {label}: {exc}")
            finally:
                del m

        if not seed_mses:
            results[key] = {"error": "all seeds failed", "label": label}
        else:
            valid_maes = [v for v in seed_maes if math.isfinite(v)]
            avg_k: dict[str, float] = {}
            if seed_ks:
                for kk in seed_ks[0]:
                    vals = [s[kk] for s in seed_ks if kk in s]
                    avg_k[kk] = round(mean(vals), 5)

            results[key] = {
                "dataset":  ds,
                "seq_len":  seq,
                "stride":   stride,
                "noise":    noise,
                "model":    mdl,
                "mse_mean": round(mean(seed_mses), 6),
                "mse_std":  round(stdev(seed_mses) if len(seed_mses) > 1 else 0.0, 6),
                "mae_mean": round(mean(valid_maes), 6) if valid_maes else None,
                "mae_std":  round(stdev(valid_maes) if len(valid_maes) > 1 else 0.0, 6),
                "seed_mses": [round(v, 6) for v in seed_mses],
                "seed_maes": [round(v, 6) for v in seed_maes],
                "kalman_stats": avg_k,
                "params":           params,
                "found_lr":         round(found_lr, 7),
                "n_train_windows":  n_windows,
                "avg_train_time_s": round(mean(seed_times), 2) if seed_times else None,
            }

        _save(OUT_FILE, results)

    elapsed_total = time.time() - sweep_start
    n_ok  = sum(1 for v in results.values() if "mse_mean" in v)
    n_err = sum(1 for v in results.values() if "error" in v or "skipped" in v)
    print(
        f"\n\nDone!  {elapsed_total / 3600:.2f}h total  |  "
        f"{n_ok} cells OK  |  {n_err} skipped/error\n"
        f"Results -> {OUT_FILE}"
    )


if __name__ == "__main__":
    main()
