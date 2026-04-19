#!/usr/bin/env python3
"""
Zaj-robusztossági sweep: noise=2.0 és noise=3.0 minden dataseten.
Méret: medium (lstm_hidden=128, heads=4, state=32), stride=1.
Cél: megmutatni hogy a KLA Kalman-szűrője strukturálisan rezisztensebb zajra.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from adapters.offline_training.ltsf_benchmark import (
    _load_dataset, _make_arima_model, _make_dlinear,
    _make_fair_mamba, _make_kla_mamba, _make_lstm,
    _train_one, _find_lr_fastai,
)

# -----------------------------------------------------------------------------
DATASETS   = ["Exchange", "ETTh1", "ETTh2", "ETTm1", "ECL", "M4Hourly", "Weather", "ETTm2"]
NOISE_VALS = [0.0, 2.0, 3.0]   # 0.0 = baseline referencia
STRIDE     = 1
SEQ_LEN    = 96
BATCH      = 64
EPOCHS     = 8
LSTM_H, F_H, F_S, K_H, K_S = 128, 4, 32, 4, 32   # medium
ARIMA_P, ARIMA_Q, DLIN_K   = 5, 3, 25

OUT_DIR  = Path(__file__).parent / "benchmark_results"
OUT_DIR.mkdir(exist_ok=True)
STAMP    = datetime.now().strftime("%Y%m%d_%H%M%S")
TXT_FILE = OUT_DIR / f"noise_sweep_{STAMP}.txt"
JSON_FILE= OUT_DIR / f"noise_sweep_{STAMP}.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lines: list[str] = []
all_results: list[dict] = []


def log(msg=""):
    lines.append(msg)
    print(msg, flush=True)


def p(n):
    return f"{n/1000:.1f}k" if n < 1_000_000 else f"{n/1e6:.2f}M"


log("=" * 100)
log(f"  Zaj-robusztossági sweep  —  noise ∈ {NOISE_VALS}")
log(f"  {datetime.now():%Y-%m-%d %H:%M:%S}  |  {device}"
    f"{'  (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else ''}")
log(f"  Méret: medium (lstm_h={LSTM_H}, heads={F_H}, state={F_S})  |  stride={STRIDE}  |  epochs={EPOCHS}")
log("=" * 100)

t0_total = time.time()
total = len(DATASETS) * len(NOISE_VALS)
idx   = 0

for ds in DATASETS:
    log(f"\n{'='*100}")
    log(f"  DATASET: {ds}")
    log(f"{'='*100}")

    # noise=0.0 baseline referencia + két zajszint közös loader párban
    ds_row: dict = {"dataset": ds, "noise_results": {}}

    for noise in NOISE_VALS:
        idx += 1
        log(f"\n  [{idx}/{total}] noise={noise}")

        try:
            train_loader, test_loader, input_dim = _load_dataset(
                ds, SEQ_LEN, BATCH, noise, STRIDE
            )
            log(f"     input_dim={input_dim}  train_windows={len(train_loader.dataset)}")
        except Exception as e:
            log(f"     [HIBA] betöltés: {e}")
            continue

        try:
            models = [
                ("lstm",       "Vanilla LSTM",  _make_lstm(input_dim, LSTM_H).to(device)),
                ("fair_mamba", "Fair Mamba",     _make_fair_mamba(input_dim, F_H, F_S).to(device)),
                ("kla_mamba",  "KLA-Mamba",      _make_kla_mamba(input_dim, K_H, K_S).to(device)),
                ("arima",      f"ARIMA({ARIMA_P},1,{ARIMA_Q})",
                    _make_arima_model(input_dim, ARIMA_P, ARIMA_Q, LSTM_H*2).to(device)),
                ("dlinear",    "DLinear",
                    _make_dlinear(input_dim, SEQ_LEN, DLIN_K).to(device)),
            ]
        except Exception as e:
            log(f"     [HIBA] modellek: {e}")
            continue

        cfg = {"lr": 1e-3, "epochs": EPOCHS,
               "lstm_hidden": LSTM_H, "fair_heads": F_H, "fair_state": F_S,
               "kla_heads": K_H, "kla_state": K_S,
               "arima_p": ARIMA_P, "arima_q": ARIMA_Q, "dlinear_kernel": DLIN_K}

        noise_row: dict = {}
        mses: dict[str, float] = {}

        for key, mname, model in models:
            try:
                try:    found_lr = _find_lr_fastai(model, train_loader, device)
                except: found_lr = 1e-3
                cfg["lr"] = found_lr
                res = _train_one(model, mname, train_loader, test_loader, cfg, device)
                mse = res["test_mse"]
                mses[key] = mse
                ks = res.get("kalman_stats", {})
                ks_str = ""
                if ks:
                    ks_str = (f"  K={ks.get('K_mean','?'):.4f}"
                              f"  A={ks.get('A_mean','?'):.4f}"
                              f"  R={ks.get('R_mean','?'):.4f}"
                              f"  gate={ks.get('res_gate_mean','?'):.4f}")
                log(f"     {mname:<22} lr={found_lr:.2e}  params={p(res['params']):<7}"
                    f"  mse={mse:.6f}  t={res['train_time_s']:.1f}s{ks_str}")
                noise_row[key] = {"mse": mse, "params": res["params"],
                                  "lr": found_lr, "epoch_losses": res["epoch_losses"],
                                  "kalman_stats": ks}
            except Exception as e:
                log(f"     {mname:<22} [HIBA] {e}")

        if "kla_mamba" in mses and "fair_mamba" in mses:
            kla_mse = mses["kla_mamba"]
            fair_mse = mses["fair_mamba"]
            diff = (kla_mse - fair_mse) / fair_mse * 100
            winner = "KLA" if kla_mse < fair_mse else "Fair"
            log(f"\n     > KLA={kla_mse:.6f}  Fair={fair_mse:.6f}"
                f"  -> {winner} nyer  ({abs(diff):.1f}%  diff)")
            noise_row["kla_vs_fair_pct"] = round(diff, 3)

        if mses:
            ranked = sorted(mses.items(), key=lambda x: x[1])
            log(f"     Ranglista: " +
                "  ".join(f"{i+1}.{k}({v:.6f})" for i, (k, v) in enumerate(ranked)))
            noise_row["ranking"] = [{"model": k, "mse": v} for k, v in ranked]

        ds_row["noise_results"][str(noise)] = noise_row

        # Közbülső mentés
        all_results.append({"dataset": ds, "noise": noise, **noise_row})
        JSON_FILE.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
        TXT_FILE.write_text("\n".join(lines), encoding="utf-8")

    # Dataset-szintű: mennyit degradálódott minden modell noise 0->2->3
    log(f"\n  -- {ds} degradációs összefoglaló --")
    r0 = ds_row["noise_results"].get("0.0", {})
    for noise in [2.0, 3.0]:
        rn = ds_row["noise_results"].get(str(noise), {})
        parts = []
        for key in ["lstm", "fair_mamba", "kla_mamba", "arima", "dlinear"]:
            m0 = r0.get(key, {}).get("mse")
            mn = rn.get(key, {}).get("mse")
            if m0 and mn:
                deg = (mn - m0) / m0 * 100
                parts.append(f"{key}:{deg:+.0f}%")
        log(f"     noise={noise}: " + "  ".join(parts))

# -----------------------------------------------------------------------------
# Globális összefoglaló
# -----------------------------------------------------------------------------
log(f"\n{'='*100}")
log("  GLOBÁLIS ZAJ-ROBUSZTOSSÁGI ÖSSZEFOGLALÓ")
log(f"{'='*100}")

# KLA vs Fair győzelmek noise-szintenként
for noise in NOISE_VALS:
    recs = [r for r in all_results
            if r.get("noise") == noise and "kla_vs_fair_pct" in r]
    kla_w = sum(1 for r in recs if r["kla_vs_fair_pct"] < -0.5)
    fair_w = sum(1 for r in recs if r["kla_vs_fair_pct"] > 0.5)
    avd = (sum(r["kla_vs_fair_pct"] for r in recs) / len(recs)) if recs else 0
    log(f"  noise={noise}: KLA nyer {kla_w}/{len(recs)}  Fair nyer {fair_w}/{len(recs)}"
        f"  átlag diff={avd:+.1f}%")

# KLA degradáció vs Fair degradáció noise 0->3
log(f"\n  Átlagos degradáció (MSE növekedés) noise 0->3:")
for key, label in [("kla_mamba", "KLA-Mamba"), ("fair_mamba", "Fair Mamba"),
                   ("lstm", "LSTM"), ("dlinear", "DLinear")]:
    degs = []
    for ds in DATASETS:
        r0 = next((r for r in all_results if r.get("dataset") == ds
                   and r.get("noise") == 0.0 and key in r), {})
        r3 = next((r for r in all_results if r.get("dataset") == ds
                   and r.get("noise") == 3.0 and key in r), {})
        m0 = r0.get(key, {}).get("mse") if isinstance(r0.get(key), dict) else None
        m3 = r3.get(key, {}).get("mse") if isinstance(r3.get(key), dict) else None
        if m0 and m3 and m0 > 0:
            degs.append((m3 - m0) / m0 * 100)
    if degs:
        log(f"    {label:<15}: {sum(degs)/len(degs):+.1f}% átlag  "
            f"(min {min(degs):+.1f}%  max {max(degs):+.1f}%)")

elapsed = time.time() - t0_total
log(f"\n  Futási idő: {elapsed/60:.1f} perc")
log("=" * 100)

TXT_FILE.write_text("\n".join(lines), encoding="utf-8")
JSON_FILE.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n  TXT: {TXT_FILE}")
print(f"  JSON: {JSON_FILE}")
