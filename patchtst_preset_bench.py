"""Run PatchTST on the 9 dashboard presets and compare to existing sweep results.

Quick-turnaround benchmark: trains PatchTST on exactly the (dataset, seq_len,
stride, noise) combinations used by the dashboard preset buttons, then prints
a side-by-side table vs. LSTM / Fair-Mamba / KCA-Mamba from sweep_results.json
(+ stride_sweep_results.json for the two stride_sweep-sourced presets).

Run: python patchtst_preset_bench.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from adapters.offline_training.ltsf_benchmark import (  # noqa: E402
    _load_dataset, _find_lr_fastai, _train_one, _make_patchtst,
)


# 9 preset cells matching dashboard PRESETS (dataset, seq_len, stride, noise, source_json)
PRESETS = [
    ("Exchange", 192, 1,  0.0, "sweep"),         # exchange_192_clean
    ("Exchange", 336, 2,  3.0, "sweep"),         # exchange_336_sigma3
    ("Exchange", 336, 2,  1.0, "sweep"),         # exchange_336_sigma1
    ("Exchange", 512, 2,  3.0, "sweep"),         # exchange_512_sigma3
    ("Exchange", 192, 1,  5.0, "stride_sweep"),  # exchange_192_sigma5
    ("Exchange", 512, 1,  5.0, "stride_sweep"),  # exchange_512_sigma5
    ("ETTm1",     96, 4,  0.0, "sweep"),         # ettm1_clean
    ("ETTm1",    512, 16, 5.0, "sweep"),         # ettm1_512_sigma5
    ("Weather",  336, 16, 5.0, "sweep"),         # weather_336_sigma5
]

EPOCHS     = 10
BATCH_SIZE = 64
ROOT       = pathlib.Path(__file__).parent
OUT_FILE   = ROOT / "patchtst_preset_results.json"


def _load_existing_sweeps() -> dict:
    """Return lookup: (ds, seq, noise, model) -> test_mse from the two sweep JSONs."""
    out: dict = {}
    sweep = json.loads((ROOT / "sweep_results.json").read_text())
    for k, v in sweep.items():
        if not isinstance(v, dict) or "test_mse" not in v:
            continue
        out[(v["dataset"], v["seq_len"], float(v["noise"]), v["model"])] = v["test_mse"]
    stride_sweep = json.loads((ROOT / "stride_sweep_results.json").read_text())
    for k, v in stride_sweep.items():
        if not isinstance(v, dict) or "test_mse" not in v:
            continue
        # key format: Exchange|seqN|sS|noise|model  —  only use stride=1 cells here
        parts = k.split("|")
        if len(parts) == 5 and parts[2] == "s1":
            ds = parts[0]
            seq = int(parts[1].removeprefix("seq"))
            noise = float(parts[3])
            mdl = parts[4]
            out.setdefault((ds, seq, noise, mdl), v["test_mse"])
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
    for i, (ds, seq, stride, noise, src) in enumerate(PRESETS, 1):
        key = f"{ds}|{seq}|s{stride}|{noise:.1f}|patchtst"
        if key in results and "test_mse" in results[key]:
            print(f"[{i}/9] {key}  (cached, skip)")
            continue

        label = f"[{i}/9] {ds} seq={seq} stride={stride} σ={noise}"
        print(label + " ...")
        try:
            train_loader, test_loader, input_dim = _load_dataset(
                ds, seq, BATCH_SIZE, noise_scale=noise, stride=stride
            )
        except Exception as exc:
            print(f"  [SKIP] data: {exc}")
            results[key] = {"error": f"data: {exc}"}
            OUT_FILE.write_text(json.dumps(results, indent=2))
            continue

        # PatchTST with ~60k-param config
        model = _make_patchtst(
            input_dim, seq_len=seq,
            patch_len=16, stride=16, d_model=48, n_heads=4, n_layers=2,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        try:
            found_lr = _find_lr_fastai(model, train_loader, device)
            found_lr = max(1e-5, min(found_lr, 5e-3))
        except Exception:
            found_lr = 1e-3

        try:
            res = _train_one(
                model, "patchtst", train_loader, test_loader,
                {"lr": found_lr, "epochs": EPOCHS}, device,
            )
        except Exception as exc:
            print(f"  [ERR] train: {exc}")
            results[key] = {"error": f"train: {exc}"}
            OUT_FILE.write_text(json.dumps(results, indent=2))
            continue

        results[key] = {
            "dataset": ds, "seq_len": seq, "stride": stride, "noise": noise,
            "model": "patchtst", "n_params": n_params, "found_lr": round(found_lr, 8),
            "test_mse": res["test_mse"], "train_time_s": res["train_time_s"],
            "peak_vram_mb": res["peak_vram_mb"],
            "source_sweep": src,
        }
        OUT_FILE.write_text(json.dumps(results, indent=2))
        print(
            f"  ✓ mse={res['test_mse']:.4f}  params={n_params:>6d}  "
            f"lr={found_lr:.2e}  t={res['train_time_s']:.1f}s"
        )

    total_min = (time.time() - t_start) / 60
    print(f"\nDone in {total_min:.1f} min.\n")

    # ── Build comparison table ────────────────────────────────────────────────
    print("=" * 100)
    print(f"{'preset':<26} {'LSTM':>10} {'Fair-M':>10} {'KCA-M':>10} {'PatchTST':>10}"
          f"  {'Δ vs KCA':>12}")
    print("-" * 100)
    for (ds, seq, stride, noise, src) in PRESETS:
        tag = f"{ds[:3]}_{seq}_s{stride}_σ{noise:.0f}"
        row = [tag]
        k_patch = f"{ds}|{seq}|s{stride}|{noise:.1f}|patchtst"
        patch = results.get(k_patch, {}).get("test_mse")
        lstm = sweeps.get((ds, seq, noise, "lstm"))
        fair = sweeps.get((ds, seq, noise, "fair_mamba"))
        kca  = sweeps.get((ds, seq, noise, "kca_mamba"))
        for v in (lstm, fair, kca, patch):
            row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "—")
        if isinstance(kca, (int, float)) and isinstance(patch, (int, float)):
            pct = (patch - kca) / kca * 100
            delta = f"{pct:+6.1f}% vs KCA"
        else:
            delta = "—"
        row.append(delta)
        print(f"{row[0]:<26} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10}"
              f"  {row[5]:>12}")

    print("=" * 100)
    print(f"\nΔ > 0  →  PatchTST rosszabb (KCA nyer)")
    print(f"Δ < 0  →  PatchTST jobb")
    print(f"\nResults saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
