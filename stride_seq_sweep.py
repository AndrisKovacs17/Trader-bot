"""
Stride × long-sequence sweep.

Combines the stride dimension (s=1, s=16) with longer seq_len
(192, 336, 512) to check whether KCA's advantage persists when
the training window is sparse (s=16) at longer context sizes.

Dataset : Exchange-Rate  (only dataset where stride effect was dramatic)
seq_len : 192, 336, 512  (T=96 already in sweep_results.json)
strides : 1, 16
noise   : 0.0, 1.0, 3.0, 5.0
Models  : lstm, fair_mamba, kla_mamba   (ARIMA skipped: deterministic, slow)
Epochs  : 10

Results saved incrementally to stride_sweep_results.json after every cell.
Run:
    python stride_seq_sweep.py
    python stride_seq_sweep.py --resume
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import argparse

# ── Sweep config ───────────────────────────────────────────────────────────────
DATASET    = "Exchange"
SEQ_LENS   = [192, 336, 512]
STRIDES    = [1, 16]
NOISES     = [0.0, 1.0, 3.0, 5.0]
EPOCHS     = 10
BATCH_SIZE = 64
LR_WARMUP  = True
OUT_FILE   = pathlib.Path(__file__).parent / "stride_sweep_results.json"

MODEL_CFGS = {
    "lstm":       {"lstm_hidden": 128},
    "fair_mamba": {"fair_heads": 4, "fair_state": 32},
    "kla_mamba":  {"kla_heads": 4,  "kla_state": 32},
}


def _cell_key(seq: int, stride: int, noise: float, model: str) -> str:
    return f"{DATASET}|seq{seq}|s{stride}|{noise:.1f}|{model}"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("PyTorch not found.", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from adapters.offline_training.ltsf_benchmark import (
        _make_lstm,
        _make_fair_mamba,
        _make_kla_mamba,
        _load_dataset,
        _find_lr_fastai,
        _train_one,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = _load_results(OUT_FILE) if args.resume else {}

    # Build cells
    cells: list[tuple] = []
    for stride in STRIDES:
        for seq in SEQ_LENS:
            for noise in NOISES:
                for mdl in MODEL_CFGS:
                    key = _cell_key(seq, stride, noise, mdl)
                    if args.resume and key in results:
                        continue
                    cells.append((seq, stride, noise, mdl, key))

    total = len(cells)
    all_cells = len(STRIDES) * len(SEQ_LENS) * len(NOISES) * len(MODEL_CFGS)
    skipped = all_cells - total
    print(f"Total cells: {all_cells}  |  skipped: {skipped}  |  to run: {total}")
    print(f"Output: {OUT_FILE}\n")

    sweep_start = time.time()
    done = 0

    for (seq, stride, noise, mdl, key) in cells:
        label = f"Exchange seq={seq} s={stride} σ={noise:.1f} {mdl}"
        _progress(done, total, sweep_start, label)

        # Load data
        t0 = time.time()
        try:
            train_loader, test_loader, input_dim = _load_dataset(
                DATASET, seq, BATCH_SIZE, noise_scale=noise, stride=stride
            )
        except Exception as exc:
            results[key] = {"error": str(exc)}
            _save_results(OUT_FILE, results)
            done += 1
            print(f"\n  [SKIP] {label}: {exc}")
            continue

        # Build model
        cfg = MODEL_CFGS[mdl]
        try:
            if mdl == "lstm":
                model = _make_lstm(input_dim, int(cfg["lstm_hidden"])).to(device)
            elif mdl == "fair_mamba":
                model = _make_fair_mamba(input_dim, int(cfg["fair_heads"]),
                                         int(cfg["fair_state"])).to(device)
            elif mdl == "kla_mamba":
                model = _make_kla_mamba(input_dim, int(cfg["kla_heads"]),
                                        int(cfg["kla_state"])).to(device)
        except Exception as exc:
            results[key] = {"error": f"model init: {exc}"}
            _save_results(OUT_FILE, results)
            done += 1
            continue

        n_params = sum(p.numel() for p in model.parameters())

        # LR finder
        if LR_WARMUP:
            try:
                found_lr = _find_lr_fastai(model, train_loader, device)
                found_lr = max(1e-5, min(found_lr, 5e-3))
            except Exception:
                found_lr = 1e-3
        else:
            found_lr = 1e-3

        # Train
        train_cfg = {"lr": found_lr, "epochs": EPOCHS}
        try:
            res = _train_one(model, mdl, train_loader, test_loader, train_cfg, device)
        except Exception as exc:
            results[key] = {"error": f"train: {exc}"}
            _save_results(OUT_FILE, results)
            done += 1
            print(f"\n  [ERR] {label}: {exc}")
            continue

        # Collect K values for KCA
        k_vals = None
        if mdl == "kla_mamba" and hasattr(model, "blocks"):
            try:
                import torch
                ks = []
                with torch.no_grad():
                    for blk in model.blocks:
                        if hasattr(blk, "kf") and hasattr(blk.kf, "k_net"):
                            # dummy forward to get K estimate
                            dummy = torch.zeros(1, seq, input_dim, device=device)
                            _, kout = blk.kf(dummy)
                            ks.append(float(kout.mean()))
                if ks:
                    k_vals = round(float(sum(ks) / len(ks)), 4)
            except Exception:
                pass

        entry = {
            "dataset":   DATASET,
            "seq_len":   seq,
            "stride":    stride,
            "noise":     noise,
            "model":     mdl,
            "n_params":  n_params,
            "found_lr":  round(found_lr, 8),
            "test_mse":  round(float(res["test_mse"]), 6),
            "train_time_s": round(float(res.get("train_time_s", 0)), 2),
        }
        if k_vals is not None:
            entry["k_mean"] = k_vals

        results[key] = entry
        _save_results(OUT_FILE, results)

        done += 1
        print()  # newline after progress bar

    _progress(done, total, sweep_start, "DONE")
    print(f"\n\nAll done! Results: {OUT_FILE}")

    # Print summary table
    print("\n" + "=" * 78)
    print(f"{'Seq':>5} {'Stride':>7} {'Noise':>6}  {'LSTM':>9}  {'Fair':>9}  {'KLA':>9}  {'KLA-LSTM%':>10}")
    print("=" * 78)
    for stride in STRIDES:
        for seq in SEQ_LENS:
            for noise in NOISES:
                row = {}
                for mdl in ["lstm", "fair_mamba", "kla_mamba"]:
                    k = _cell_key(seq, stride, noise, mdl)
                    if k in results and "test_mse" in results[k]:
                        row[mdl] = results[k]["test_mse"]
                lstm_m = row.get("lstm", float("nan"))
                fair_m = row.get("fair_mamba", float("nan"))
                kla_m  = row.get("kla_mamba", float("nan"))
                vs_lstm = (lstm_m - kla_m) / lstm_m * 100 if lstm_m else float("nan")
                print(
                    f"{seq:>5} {stride:>7} {noise:>6.1f}  "
                    f"{lstm_m:>9.4f}  {fair_m:>9.4f}  {kla_m:>9.4f}  {vs_lstm:>+10.1f}%"
                )
        print("-" * 78)


if __name__ == "__main__":
    main()
