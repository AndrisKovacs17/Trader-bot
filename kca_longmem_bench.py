"""Compare original KCA-Mamba vs Decoupled-A KCA on:
  - 9 standard dashboard presets  (same as before)
  - 3 long-memory presets: ETTh1/ETTh2 seq=720 (napi+heti ciklus), Weather seq=720

ETTh1/ETTh2 hourly: napi ciklus = 24 lepes, heti = 168 lepes
Weather 10min:      napi ciklus = 144 lepes -> seq=720 = 5 nap

A decoupled-A modell: A = exp(-softplus(log_A)), fuggetlen K-tol.
Az eredeti KCA: A = clamp(1-K, 0.01, 0.99).

Run: python kca_longmem_bench.py
"""
from __future__ import annotations
import json, pathlib, sys, time
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from adapters.offline_training.ltsf_benchmark import (
    _load_dataset, _find_lr_fastai, _train_one, _make_kca_mamba,
)

ROOT     = pathlib.Path(__file__).parent
OUT_FILE = ROOT / "kca_longmem_results.json"

PRESETS = [
    ("Exchange", 192,  1,  0.0, "sweep"),
    ("Exchange", 336,  2,  3.0, "sweep"),
    ("Exchange", 336,  2,  1.0, "sweep"),
    ("Exchange", 512,  2,  3.0, "sweep"),
    ("Exchange", 192,  1,  5.0, "stride_sweep"),
    ("Exchange", 512,  1,  5.0, "stride_sweep"),
    ("ETTm1",     96,  4,  0.0, "sweep"),
    ("ETTm1",    512, 16,  5.0, "sweep"),
    ("Weather",  336, 16,  5.0, "sweep"),
    # long-memory: napi + heti ciklus
    ("ETTh1",    720,  1,  0.0, "longmem"),
    ("ETTh2",    720,  1,  0.0, "longmem"),
    ("Weather",  720,  1,  0.0, "longmem"),
]

EPOCHS     = 10
BATCH_SIZE = 64
MODEL_NAME = "kca_decoupled"


def _load_existing_sweeps():
    out = {}
    try:
        sweep = json.loads((ROOT / "sweep_results.json").read_text())
        for _, v in sweep.items():
            if isinstance(v, dict) and "test_mse" in v:
                out[(v["dataset"], v["seq_len"], float(v["noise"]), v["model"])] = v["test_mse"]
    except Exception:
        pass
    try:
        ss = json.loads((ROOT / "stride_sweep_results.json").read_text())
        for k, v in ss.items():
            if not (isinstance(v, dict) and "test_mse" in v): continue
            parts = k.split("|")
            if len(parts) == 5 and parts[2] == "s1":
                ds = parts[0]; seq = int(parts[1].removeprefix("seq"))
                noise = float(parts[3]); mdl = parts[4]
                out.setdefault((ds, seq, noise, mdl), v["test_mse"])
    except Exception:
        pass
    return out


def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sweeps  = _load_existing_sweeps()
    results = {}
    if OUT_FILE.exists():
        try: results = json.loads(OUT_FILE.read_text())
        except Exception: pass

    total = len(PRESETS); t0 = time.time()
    for i, (ds, seq, stride, noise, src) in enumerate(PRESETS, 1):
        key = f"{ds}|{seq}|s{stride}|{noise:.1f}|{MODEL_NAME}"
        if key in results and "test_mse" in results[key]:
            print(f"[{i:>2}/{total}] {key}  (cached, skip)"); continue
        try:
            train_loader, test_loader, input_dim = _load_dataset(
                ds, seq, BATCH_SIZE, noise_scale=noise, stride=stride)
        except Exception as exc:
            print(f"[DATA-SKIP] {ds}|{seq}|{noise}: {exc}"); continue

        model = _make_kca_mamba(input_dim, heads=4, d_state=32).to(device)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"[{i:>2}/{total}] {ds} seq={seq} s={stride} sigma={noise} :: "
              f"KCA-decoupled ({n_par} par) ...", flush=True)

        try:
            lr = max(1e-5, min(_find_lr_fastai(model, train_loader, device), 5e-3))
        except Exception:
            lr = 1e-3

        try:
            res = _train_one(model, MODEL_NAME, train_loader, test_loader,
                             {"lr": lr, "epochs": EPOCHS}, device)
        except Exception as exc:
            print(f"       [ERR] {exc}")
            results[key] = {"error": str(exc)}
            OUT_FILE.write_text(json.dumps(results, indent=2)); continue

        results[key] = {
            "dataset": ds, "seq_len": seq, "stride": stride, "noise": noise,
            "n_params": n_par, "found_lr": round(lr, 8),
            "test_mse": res["test_mse"], "train_time_s": res["train_time_s"],
        }
        OUT_FILE.write_text(json.dumps(results, indent=2))
        print(f"       mse={res['test_mse']:.4f}  lr={lr:.2e}  t={res['train_time_s']:.1f}s")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min.\n")

    print("=" * 88)
    print(f"{'preset':<24} {'KCA-orig':>10} {'KCA-decoupled':>14}  {'Delta%':>7}  mem-depth")
    print("-" * 88)
    for (ds, seq, stride, noise, src) in PRESETS:
        tag  = f"{ds[:3]}_{seq}_s{stride}_s{noise:.0f}"
        orig = sweeps.get((ds, seq, noise, "kca_mamba"))
        key  = f"{ds}|{seq}|s{stride}|{noise:.1f}|{MODEL_NAME}"
        new  = results.get(key, {}).get("test_mse")
        mem  = "~10k steps" if seq >= 512 else "~1k steps"
        so   = f"{orig:.4f}" if isinstance(orig, float) else "-"
        sn   = f"{new:.4f}"  if isinstance(new,  float) else "-"
        if isinstance(orig, float) and isinstance(new, float):
            sd = f"{(new-orig)/orig*100:+.1f}%"
        else:
            sd = "-"
        print(f"{tag:<24} {so:>10} {sn:>14}  {sd:>7}  {mem}")
    print("=" * 88)
    print("Delta < 0 = decoupled JOBB")
    print(f"Results: {OUT_FILE}")


if __name__ == "__main__":
    main()
