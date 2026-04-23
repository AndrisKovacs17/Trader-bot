"""
Long-sequence noise robustness sweep.

Datasets : ETTm1 (15-min), Exchange (daily), Weather (10-min)
seq_len  : 96, 192, 336, 512
noise    : 0.0, 1.0, 3.0, 5.0
Models   : LSTM, Fair Mamba, KCA-Mamba, ARIMA(5,1,3)
Epochs   : 10 (more than default for better convergence at long seq_len)

Results saved incrementally to sweep_results.json after every cell.
Run: python long_seq_sweep.py
     python long_seq_sweep.py --resume   (skip already-done cells)
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import argparse

# ── Sweep config ───────────────────────────────────────────────────────────────
DATASETS   = ["ETTm1", "Exchange", "Weather"]
SEQ_LENS   = [96, 192, 336, 512]
NOISES     = [0.0, 1.0, 3.0, 5.0]
EPOCHS     = 10
BATCH_SIZE = 64
LR_WARMUP  = True     # use FastAI LR finder; False = fixed 1e-3
OUT_FILE   = pathlib.Path(__file__).parent / "sweep_results.json"

MODEL_CFGS = {
    "lstm":       {"lstm_hidden": 128},
    "fair_mamba": {"fair_heads": 4, "fair_state": 32},
    "kca_mamba":  {"kca_heads": 4,  "kca_state": 32},
    "arima":      {"arima_p": 5, "arima_q": 3, "lstm_hidden": 256},
}

# stride per (dataset, seq_len) — keeps batch count reasonable
STRIDE = {
    ("ETTm1",    96): 4,
    ("ETTm1",   192): 8,
    ("ETTm1",   336): 12,
    ("ETTm1",   512): 16,
    ("Exchange",  96): 1,
    ("Exchange", 192): 1,
    ("Exchange", 336): 2,
    ("Exchange", 512): 2,
    ("Weather",   96): 8,
    ("Weather",  192): 12,
    ("Weather",  336): 16,
    ("Weather",  512): 24,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cell_key(ds: str, seq: int, noise: float, model: str) -> str:
    return f"{ds}|{seq}|{noise:.1f}|{model}"


def _load_results(path: pathlib.Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def _save_results(path: pathlib.Path, results: dict) -> None:
    path.write_text(json.dumps(results, indent=2))


def _progress(done: int, total: int, start_t: float, label: str) -> None:
    elapsed = time.time() - start_t
    eta = (elapsed / done * (total - done)) if done > 0 else 0
    bar_w = 30
    filled = int(bar_w * done / total)
    bar = "█" * filled + "░" * (bar_w - filled)
    print(
        f"\r[{bar}] {done}/{total}  "
        f"elapsed {elapsed/60:.1f}m  ETA {eta/60:.1f}m  | {label}",
        end="", flush=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip cells already present in sweep_results.json")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="Subset of datasets to run")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=SEQ_LENS)
    parser.add_argument("--noises", nargs="+", type=float, default=NOISES)
    parser.add_argument("--models", nargs="+", default=list(MODEL_CFGS.keys()))
    args = parser.parse_args()

    # ── Import GPU/torch-dependent code ───────────────────────────────────────
    try:
        import torch
    except ImportError:
        print("PyTorch not found. Install with: pip install torch", file=sys.stderr)
        sys.exit(1)

    # Reuse factories from the existing benchmark (avoids duplication / divergence)
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from adapters.offline_training.ltsf_benchmark import (
        _make_lstm,
        _make_fair_mamba,
        _make_kca_mamba,
        _make_arima_model,
        _load_dataset,
        _find_lr_fastai,
        _train_one,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = _load_results(OUT_FILE) if args.resume else {}

    # Build cell list
    cells: list[tuple] = []
    for ds in args.datasets:
        for seq in args.seq_lens:
            for noise in args.noises:
                for mdl in args.models:
                    key = _cell_key(ds, seq, noise, mdl)
                    if args.resume and key in results:
                        continue
                    cells.append((ds, seq, noise, mdl, key))

    total = len(cells)
    if total == 0:
        print("Nothing to run (all cells already done). Use --resume or delete sweep_results.json.")
        return

    # Precount for stats
    all_cells = sum(1 for ds in args.datasets for _ in args.seq_lens
                    for _ in args.noises for _ in args.models)
    skip = all_cells - total
    print(f"Total cells: {all_cells}  |  skipped: {skip}  |  to run: {total}")
    print(f"Output: {OUT_FILE}\n")

    sweep_start = time.time()
    done = 0

    for (ds, seq, noise, mdl, key) in cells:
        label = f"{ds} seq={seq:3d} σ={noise:.1f} {mdl}"
        _progress(done, total, sweep_start, label)

        # ── Load data ──────────────────────────────────────────────────────────
        stride = STRIDE.get((ds, seq), 8)
        t0 = time.time()
        try:
            train_loader, test_loader, input_dim = _load_dataset(
                ds, seq, BATCH_SIZE, noise_scale=noise, stride=stride
            )
        except Exception as exc:
            results[key] = {"error": str(exc)}
            _save_results(OUT_FILE, results)
            done += 1
            print(f"\n  [SKIP] {label}: {exc}")
            continue

        load_s = round(time.time() - t0, 2)

        # ── Build model ────────────────────────────────────────────────────────
        cfg = MODEL_CFGS[mdl]
        try:
            if mdl == "lstm":
                model = _make_lstm(input_dim, int(cfg["lstm_hidden"])).to(device)
            elif mdl == "fair_mamba":
                model = _make_fair_mamba(input_dim,
                                         int(cfg["fair_heads"]),
                                         int(cfg["fair_state"])).to(device)
            elif mdl == "kca_mamba":
                model = _make_kca_mamba(input_dim,
                                        int(cfg["kca_heads"]),
                                        int(cfg["kca_state"])).to(device)
            elif mdl == "arima":
                model = _make_arima_model(input_dim,
                                          int(cfg["arima_p"]),
                                          int(cfg["arima_q"]),
                                          int(cfg["lstm_hidden"])).to(device)
            else:
                raise ValueError(f"Unknown model: {mdl}")
        except Exception as exc:
            results[key] = {"error": f"model init: {exc}"}
            _save_results(OUT_FILE, results)
            done += 1
            continue

        n_params = sum(p.numel() for p in model.parameters())

        # ── LR finder ─────────────────────────────────────────────────────────
        if LR_WARMUP:
            try:
                found_lr = _find_lr_fastai(model, train_loader, device)
                found_lr = max(1e-5, min(found_lr, 5e-3))  # safety clamp
            except Exception:
                found_lr = 1e-3
        else:
            found_lr = 1e-3

        # ── Train + test ───────────────────────────────────────────────────────
        train_cfg = {
            "lr":     found_lr,
            "epochs": EPOCHS,
        }
        try:
            res = _train_one(model, mdl, train_loader, test_loader, train_cfg, device)
        except Exception as exc:
            results[key] = {"error": f"train: {exc}"}
            _save_results(OUT_FILE, results)
            done += 1
            print(f"\n  [ERR] {label}: {exc}")
            continue

        # ── Store ──────────────────────────────────────────────────────────────
        entry = {
            "dataset":   ds,
            "seq_len":   seq,
            "noise":     noise,
            "model":     mdl,
            "n_params":  n_params,
            "found_lr":  round(found_lr, 8),
            "load_s":    load_s,
            "test_mse":  res["test_mse"],
            "train_time_s": res["train_time_s"],
            "peak_vram_mb": res["peak_vram_mb"],
            "epoch_losses": res["epoch_losses"],
            "kalman_stats": res.get("kalman_stats", {}),
        }
        results[key] = entry
        _save_results(OUT_FILE, results)

        done += 1
        final_loss = res["epoch_losses"][-1] if res["epoch_losses"] else float("nan")
        print(f"\n  [OK] {label}  mse={res['test_mse']:.4f}  ep_loss={final_loss:.4f}"
              f"  lr={found_lr:.2e}  t={res['train_time_s']:.0f}s")

    print(f"\n\nSweep done. {done} cells in {(time.time()-sweep_start)/60:.1f} min")
    print(f"Results: {OUT_FILE}")

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n=== MSE Summary (test_mse) ===")
    print(f"{'Dataset':<10} {'seq':>4} {'noise':>6}  "
          + "  ".join(f"{m:>12}" for m in args.models))
    print("-" * (10 + 4 + 6 + 3 + len(args.models) * 14))

    for ds in args.datasets:
        for seq in args.seq_lens:
            for noise in args.noises:
                row = f"{ds:<10} {seq:>4} {noise:>6.1f}  "
                for mdl in args.models:
                    key = _cell_key(ds, seq, noise, mdl)
                    v = results.get(key, {})
                    if "error" in v:
                        row += f"{'ERR':>12}  "
                    elif "test_mse" in v:
                        row += f"{v['test_mse']:>12.5f}  "
                    else:
                        row += f"{'---':>12}  "
                print(row)
            print()


if __name__ == "__main__":
    main()
