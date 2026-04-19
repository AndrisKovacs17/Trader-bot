#!/usr/bin/env python3
"""
Gyors teszt: ~10 perc, előtérben fut, eredmény azonnal látható + fájlba menti.
3 dataset × 2 stride × 4 zajszint × 3 modell
"""
from __future__ import annotations
import json, sys, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch
from adapters.offline_training.ltsf_benchmark import (
    _load_dataset, _make_fair_mamba, _make_kla_mamba, _make_lstm, _train_one,
)

# -- Konfig --------------------------------------------------------------------
DATASETS = ["Exchange", "ETTh1", "ETTh2"]
STRIDES  = [1, 16]
NOISES   = [0.0, 1.0, 3.0, 5.0]
EPOCHS   = 5
LR       = 3e-3
LSTM_H, HEADS, STATE = 128, 4, 32   # medium

OUT_DIR  = Path(__file__).parent / "benchmark_results"
OUT_DIR.mkdir(exist_ok=True)
STAMP    = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_TXT  = OUT_DIR / f"quick_{STAMP}.txt"
OUT_JSON = OUT_DIR / f"quick_{STAMP}.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lines: list[str] = []
results: list[dict] = []

def log(msg=""):
    lines.append(msg); print(msg, flush=True)

def save():
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

def p(n): return f"{n/1000:.1f}k"

log("="*90)
log(f"  Gyors zaj-robusztossági teszt  |  {datetime.now():%Y-%m-%d %H:%M}  |  {device}")
log(f"  Datasets: {DATASETS}  |  Strides: {STRIDES}  |  Noise: {NOISES}  |  Epochs: {EPOCHS}")
log("="*90)

t_start = time.time()
total = len(DATASETS) * len(STRIDES) * len(NOISES)
idx = 0

for ds in DATASETS:
    for stride in STRIDES:
        log(f"\n{'-'*90}")
        log(f"  {ds}  |  stride={stride}")
        log(f"{'-'*90}")

        # Egyszer töltjük be noise=0-val (strukturálisan ugyanaz, zaj a Datasetben van)
        # Minden noise-hoz külön loadert kell, de az adat letöltés csak egyszer fut
        noise_mses: dict[str, dict[str, float]] = {}   # model -> noise -> mse

        for noise in NOISES:
            idx += 1
            tag = f"{ds}|s={stride}|n={noise}"
            log(f"\n  [{idx}/{total}] noise={noise:.1f}")

            try:
                train_loader, test_loader, input_dim = _load_dataset(
                    ds, 96, 64, noise, stride
                )
            except Exception as e:
                log(f"    [HIBA] {e}"); continue

            models = [
                ("lstm",  "LSTM",      _make_lstm(input_dim, LSTM_H).to(device)),
                ("fair",  "Fair",      _make_fair_mamba(input_dim, HEADS, STATE).to(device)),
                ("kla",   "KLA",       _make_kla_mamba(input_dim, HEADS, STATE).to(device)),
            ]

            cfg = {"lr": LR, "epochs": EPOCHS,
                   "lstm_hidden": LSTM_H, "fair_heads": HEADS, "fair_state": STATE,
                   "kla_heads": HEADS, "kla_state": STATE}

            row: dict = {"dataset": ds, "stride": stride, "noise": noise, "models": {}}
            mses: dict[str, float] = {}

            for key, mname, model in models:
                try:
                    res = _train_one(model, mname, train_loader, test_loader, cfg, device)
                    mse = res["test_mse"]
                    mses[key] = mse
                    ks = res.get("kalman_stats", {})
                    ks_str = ""
                    if ks:
                        ks_str = (f"  K={ks.get('K_mean',0):.3f}"
                                  f"  A={ks.get('A_mean',0):.3f}"
                                  f"  R={ks.get('R_mean',0):.3f}")
                    log(f"    {mname:<6}  params={p(res['params']):<6}"
                        f"  mse={mse:.6f}  t={res['train_time_s']:.1f}s"
                        f"  lossv{res['epoch_losses'][-1]:.4f}{ks_str}")
                    row["models"][key] = {"mse": mse, "params": res["params"],
                                          "losses": res["epoch_losses"], "kalman": ks}
                    noise_mses.setdefault(key, {})[noise] = mse
                except Exception as e:
                    log(f"    {mname:<6}  [HIBA] {e}")

            # KLA vs Fair
            if "kla" in mses and "fair" in mses:
                diff = (mses["kla"] - mses["fair"]) / mses["fair"] * 100
                winner = "KLA [OK]" if diff < 0 else "Fair [OK]"
                row["kla_vs_fair_pct"] = round(diff, 2)
                log(f"\n    > KLA={mses['kla']:.6f}  Fair={mses['fair']:.6f}"
                    f"  -> {winner}  ({abs(diff):.1f}%)")

            ranked = sorted(mses.items(), key=lambda x: x[1])
            log("    Sorrend: " + "  ".join(f"{i+1}.{k}={v:.6f}" for i,(k,v) in enumerate(ranked)))

            results.append(row)
            save()

        # Stride-szintű degradáció táblázat
        log(f"\n  +- Degradáció noise növekedéskor (stride={stride}) -----------------+")
        log(f"  | {'modell':<6} " + "  ".join(f"noise={n:<4}" for n in NOISES) + "  |")
        for key in ["lstm","fair","kla"]:
            vals = [noise_mses.get(key, {}).get(n) for n in NOISES]
            vals_str = "  ".join(f"{v:.6f}" if v else "   n/a  " for v in vals)
            # degradáció 0->5
            v0, v5 = noise_mses.get(key,{}).get(0.0), noise_mses.get(key,{}).get(5.0)
            deg = f"  +{(v5-v0)/v0*100:.0f}%" if v0 and v5 else ""
            log(f"  | {key:<6} {vals_str}{deg}  |")
        log(f"  +-------------------------------------------------------------------+")

# -- Végső összefoglaló --------------------------------------------------------
log(f"\n{'='*90}")
log("  ÖSSZEFOGLALÓ: KLA vs Fair nyerések zajszintenként")
log(f"{'='*90}")
for noise in NOISES:
    recs = [r for r in results if r["noise"] == noise and "kla_vs_fair_pct" in r]
    kla_w = sum(1 for r in recs if r["kla_vs_fair_pct"] < -0.5)
    fair_w = sum(1 for r in recs if r["kla_vs_fair_pct"] > 0.5)
    avg = sum(r["kla_vs_fair_pct"] for r in recs) / len(recs) if recs else 0
    log(f"  noise={noise:.1f}  ->  KLA nyer {kla_w}/{len(recs)}  Fair nyer {fair_w}/{len(recs)}"
        f"  |  átlag KLA-Fair: {avg:+.1f}%")

# Átlagos degradáció modellenkét
log(f"\n  Átlagos MSE-növekedés noise 0->5:")
for key, label in [("kla","KLA-Mamba"),("fair","Fair Mamba"),("lstm","LSTM")]:
    degs = []
    for r0 in [r for r in results if r["noise"]==0.0]:
        m0 = r0["models"].get(key,{}).get("mse")
        r5 = next((r for r in results if r["dataset"]==r0["dataset"]
                   and r["stride"]==r0["stride"] and r["noise"]==5.0), None)
        m5 = r5["models"].get(key,{}).get("mse") if r5 else None
        if m0 and m5 and m0>0:
            degs.append((m5-m0)/m0*100)
    if degs:
        log(f"  {label:<14}: átlag {sum(degs)/len(degs):+.0f}%"
            f"  (min {min(degs):+.0f}%  max {max(degs):+.0f}%)")

log(f"\n  Futási idő: {(time.time()-t_start)/60:.1f} perc")
log(f"  Eredmény: {OUT_TXT}")
save()
