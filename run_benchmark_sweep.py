#!/usr/bin/env python3
"""
Comprehensive benchmark sweep: minden dataset × stride × modell-méret kombináció.
Eredmények: benchmark_results/sweep_<timestamp>.txt és sweep_<timestamp>.json

Cél: megmutatni hogy a KLA-Mamba stabilabb/jobb mint a Fair Mamba azonos param-számon,
különböző adatstruktúrákon és ablak-átfedési szinteken.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from adapters.offline_training.ltsf_benchmark import (
    _load_dataset,
    _make_arima_model,
    _make_dlinear,
    _make_fair_mamba,
    _make_kla_mamba,
    _make_lstm,
    _train_one,
    _find_lr_fastai,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sweep konfiguráció
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = [
    "Exchange",   # 8 dim,   ~7600 lépés  – deviza, lassú drift
    "ETTh1",      # 7 dim,   ~17k lépés   – villamoshőmérs., erős szezonalitás
    "ETTh2",      # 7 dim,   ~17k lépés   – zajosabb ETT variáns
    "ETTm1",      # 7 dim,   ~69k lépés   – 15-perces, sok adat
    "ECL",        # 321 dim, ~26k lépés   – elektromos fogyasztás, nagy dim
    "M4Hourly",   # 1 dim,   ~350k lépés  – 414 összefűzött sorozat, komplex
    "Weather",    # 21 dim,  ~52k lépés   – időjárás, nem-stacionárius
    "ETTm2",      # 7 dim,   ~69k lépés   – 15-perces ETT variáns
]

STRIDES = [1, 8, 32]          # 1=max átfedés  8=közepes  32=szinte független ablakok
SEQ_LEN   = 96
BATCH     = 64
EPOCHS    = 8                  # alapos tanuláshoz
NOISE     = 0.0                # tiszta adaton futunk (zaj teszt külön)
ARIMA_P   = 5
ARIMA_Q   = 3
DLINEAR_K = 25

# Modell méretek – kb. ~50k param ECL (321 dim)-en kívül, kis dimenzión ~10-20k
# Az arányos összehasonlításhoz minden modell HEADS * STATE ≈ ugyanolyan E-vel épül
MODEL_CONFIGS = {
    # név:  (lstm_hidden, fair_heads, fair_state, kla_heads, kla_state)
    "small":  ( 64,  2, 16,  2, 16),   # ~5-15k param
    "medium": (128,  4, 32,  4, 32),   # ~20-60k param  ← fő config
    "large":  (256,  4, 64,  4, 64),   # ~80-200k param
}

# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR = Path(__file__).parent / "benchmark_results"
OUT_DIR.mkdir(exist_ok=True)
STAMP      = datetime.now().strftime("%Y%m%d_%H%M%S")
TXT_FILE   = OUT_DIR / f"sweep_{STAMP}.txt"
JSON_FILE  = OUT_DIR / f"sweep_{STAMP}.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_results: list[dict] = []
summary_lines: list[str] = []


def log(msg: str = "", also_print: bool = True) -> None:
    summary_lines.append(msg)
    if also_print:
        print(msg, flush=True)


def param_str(n: int) -> str:
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f"{n/1000:.1f}k"
    return f"{n/1_000_000:.2f}M"


def pct_diff(a: float, b: float) -> str:
    """How much worse is b vs a? (+X% means b is X% higher MSE = worse)."""
    if a == 0:
        return "n/a"
    d = (b - a) / a * 100
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}%"


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

log("=" * 100)
log(f"  KLA-Mamba vs Fair Mamba vs LSTM vs ARIMA vs DLinear — Teljes Sweep")
log(f"  Dátum   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"  Eszköz  : {device}  (CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'nincs'})")
log(f"  Dataset : {', '.join(DATASETS)}")
log(f"  Stride-k: {STRIDES}")
log(f"  Méret   : {list(MODEL_CONFIGS.keys())}")
log(f"  Epoch   : {EPOCHS}  |  seq_len={SEQ_LEN}  |  batch={BATCH}")
log("=" * 100)

total_runs = len(DATASETS) * len(STRIDES) * len(MODEL_CONFIGS)
run_idx = 0
t_sweep_start = time.time()

for ds_name in DATASETS:
    log(f"\n{'━'*100}")
    log(f"  DATASET: {ds_name}")
    log(f"{'━'*100}")

    for stride in STRIDES:
        log(f"\n  ── Stride = {stride} ──")

        # Adatbetöltés egyszer stride-onként (minden modell-méret ugyanazt látja)
        try:
            train_loader, test_loader, input_dim = _load_dataset(
                ds_name, SEQ_LEN, BATCH, NOISE, stride
            )
            n_train = len(train_loader.dataset)
            n_test  = len(test_loader.dataset)
            log(f"     input_dim={input_dim}  train_windows={n_train}  test_windows={n_test}  "
                f"batches/epoch={len(train_loader)}")
        except Exception as e:
            log(f"     [HIBA] Adatbetöltés sikertelen: {e}")
            continue

        for size_name, (lstm_h, f_heads, f_state, k_heads, k_state) in MODEL_CONFIGS.items():
            run_idx += 1
            run_tag = f"{ds_name}|stride={stride}|{size_name}"
            log(f"\n  [{run_idx}/{total_runs}] {run_tag}")

            # Modell-lista felépítése
            try:
                arima_hidden = lstm_h * 2
                models_to_run = [
                    ("lstm",        "Vanilla LSTM",
                        _make_lstm(input_dim, lstm_h).to(device)),
                    ("fair_mamba",  "Fair Mamba",
                        _make_fair_mamba(input_dim, f_heads, f_state).to(device)),
                    ("kla_mamba",   "KLA-Mamba",
                        _make_kla_mamba(input_dim, k_heads, k_state).to(device)),
                    ("arima",       f"ARIMA({ARIMA_P},1,{ARIMA_Q})",
                        _make_arima_model(input_dim, ARIMA_P, ARIMA_Q, arima_hidden).to(device)),
                    ("dlinear",     "DLinear",
                        _make_dlinear(input_dim, SEQ_LEN, DLINEAR_K).to(device)),
                ]
            except Exception as e:
                log(f"     [HIBA] Modellek felépítése: {e}")
                continue

            # Param-számok kiírása
            p_parts = []
            for key, mname, m in models_to_run:
                pc = sum(p.numel() for p in m.parameters())
                p_parts.append(f"{mname}={param_str(pc)}")
            log(f"     Paramszámok: {' | '.join(p_parts)}")

            run_record = {
                "dataset": ds_name, "stride": stride, "model_size": size_name,
                "input_dim": input_dim, "seq_len": SEQ_LEN,
                "n_train_windows": n_train, "n_test_windows": n_test,
                "results": {}
            }

            cfg_for_train = {
                "lr": 1e-3, "epochs": EPOCHS,
                "lstm_hidden": lstm_h,
                "fair_heads": f_heads, "fair_state": f_state,
                "kla_heads": k_heads,  "kla_state": k_state,
                "arima_p": ARIMA_P,    "arima_q": ARIMA_Q,
                "dlinear_kernel": DLINEAR_K,
            }

            model_mses: dict[str, float] = {}

            for key, mname, model in models_to_run:
                t0 = time.time()
                try:
                    # LR finder
                    try:
                        found_lr = _find_lr_fastai(model, train_loader, device)
                    except Exception:
                        found_lr = 1e-3
                    cfg_for_train["lr"] = found_lr

                    res = _train_one(model, mname, train_loader, test_loader,
                                     cfg_for_train, device)
                    mse     = res["test_mse"]
                    params  = res["params"]
                    tr_time = res["train_time_s"]
                    vram    = res["peak_vram_mb"]
                    losses  = res["epoch_losses"]
                    model_mses[key] = mse

                    # Kalman stats (KLA only)
                    ks = res.get("kalman_stats", {})
                    ks_str = ""
                    if ks:
                        ks_str = (f"  K_mean={ks.get('K_mean', '?'):.4f}"
                                  f"  A_mean={ks.get('A_mean', '?'):.4f}"
                                  f"  R_mean={ks.get('R_mean', '?'):.4f}"
                                  f"  gate={ks.get('res_gate_mean', '?'):.4f}")

                    log(f"     {mname:<22} lr={found_lr:.2e}  params={param_str(params):<7}"
                        f"  mse={mse:.6f}  t={tr_time:.1f}s  vram={vram:.0f}MB"
                        f"  losses=[{', '.join(f'{v:.4f}' for v in losses)}]"
                        f"{ks_str}")

                    run_record["results"][key] = {
                        "model_name": mname, "params": params, "found_lr": found_lr,
                        "test_mse": mse, "train_time_s": tr_time,
                        "peak_vram_mb": vram, "epoch_losses": losses,
                        "kalman_stats": ks,
                    }

                except Exception as e:
                    log(f"     {mname:<22} [HIBA] {e}")
                    run_record["results"][key] = {"error": str(e)}

            # KLA vs Fair Mamba összehasonlítás
            if "kla_mamba" in model_mses and "fair_mamba" in model_mses:
                kla  = model_mses["kla_mamba"]
                fair = model_mses["fair_mamba"]
                diff_pct = (kla - fair) / fair * 100
                winner = "KLA" if kla < fair else "Fair"
                margin = abs(diff_pct)
                log(f"\n     ► KLA vs Fair Mamba: KLA MSE={kla:.6f}  Fair MSE={fair:.6f}"
                    f"  → {winner} nyer  ({margin:.1f}% különbség)")
                run_record["kla_vs_fair_pct"] = round(diff_pct, 3)

            # Ranglista ezen a configuon
            if model_mses:
                ranked = sorted(model_mses.items(), key=lambda x: x[1])
                rank_str = "  ".join(f"{i+1}.{k}({v:.6f})" for i, (k, v) in enumerate(ranked))
                log(f"     Ranglista: {rank_str}")
                run_record["ranking"] = [{"model": k, "mse": v} for k, v in ranked]

            all_results.append(run_record)

            # Menet közbeni mentés — ha a folyamat megszakad, ne vesszük el az adatokat
            _partial = [{**r, "results": {k: {kk: vv for kk, vv in v.items() if kk != "sample"}
                         for k, v in r.get("results", {}).items()}} for r in all_results]
            JSON_FILE.write_text(json.dumps(_partial, indent=2, ensure_ascii=False), encoding="utf-8")
            TXT_FILE.write_text("\n".join(summary_lines), encoding="utf-8")

    # Dataset-szintű összefoglaló
    ds_records = [r for r in all_results if r["dataset"] == ds_name]
    log(f"\n  ── {ds_name} összefoglaló ──")
    for size_name in MODEL_CONFIGS:
        recs_sz = [r for r in ds_records if r["model_size"] == size_name]
        for stride in STRIDES:
            recs = [r for r in recs_sz if r["stride"] == stride]
            if not recs:
                continue
            rec = recs[0]
            kv  = rec.get("kla_vs_fair_pct")
            rnk = rec.get("ranking", [])
            best = rnk[0]["model"] if rnk else "?"
            sign = ("KLA jobb" if (kv is not None and kv < 0)
                    else "Fair jobb" if (kv is not None and kv > 0) else "?")
            log(f"     {size_name:<8} stride={stride:<4} best={best:<14} "
                f"kla_vs_fair={kv:+.1f}% ({sign})" if kv is not None
                else f"     {size_name:<8} stride={stride:<4} best={best}")

# ─────────────────────────────────────────────────────────────────────────────
# Globális összefoglaló táblázat
# ─────────────────────────────────────────────────────────────────────────────

log("\n" + "=" * 100)
log("  GLOBÁLIS ÖSSZEFOGLALÓ")
log("=" * 100)

# KLA vs Fair győzelmi mérleg
kla_wins = fair_wins = ties = 0
kla_margins: list[float] = []
for r in all_results:
    kv = r.get("kla_vs_fair_pct")
    if kv is None:
        continue
    if kv < -0.5:
        kla_wins += 1
        kla_margins.append(-kv)
    elif kv > 0.5:
        fair_wins += 1
    else:
        ties += 1

total_matchups = kla_wins + fair_wins + ties
log(f"\n  KLA vs Fair Mamba győzelmek (minden dataset × stride × méret):")
log(f"    KLA  nyer: {kla_wins}/{total_matchups} "
    f"({100*kla_wins/max(total_matchups,1):.0f}%)")
log(f"    Fair nyer: {fair_wins}/{total_matchups} "
    f"({100*fair_wins/max(total_matchups,1):.0f}%)")
log(f"    Döntetlen: {ties}/{total_matchups}")
if kla_margins:
    log(f"    KLA átlagos előnye (ahol nyer): {sum(kla_margins)/len(kla_margins):.1f}%")

# Per-dataset KLA score
log(f"\n  Per-dataset KLA győzelmi arány:")
for ds in DATASETS:
    recs = [r for r in all_results if r["dataset"] == ds and r.get("kla_vs_fair_pct") is not None]
    kw = sum(1 for r in recs if r["kla_vs_fair_pct"] < -0.5)
    log(f"    {ds:<12}: {kw}/{len(recs)} győzelem")

# Ranglista összesítés: hány 1. hely per modell
first_place: dict[str, int] = {}
for r in all_results:
    rnk = r.get("ranking", [])
    if rnk:
        m = rnk[0]["model"]
        first_place[m] = first_place.get(m, 0) + 1

log(f"\n  #1 helyek összesen (minden konfig):")
for mname, cnt in sorted(first_place.items(), key=lambda x: -x[1]):
    log(f"    {mname:<15}: {cnt} × első hely")

# Stride-érzékenység: KLA vs Fair különbség stride szerint
log(f"\n  Stride-érzékenység (KLA vs Fair átlag eltérés):")
for s in STRIDES:
    recs = [r for r in all_results
            if r["stride"] == s and r.get("kla_vs_fair_pct") is not None]
    if recs:
        avg = sum(r["kla_vs_fair_pct"] for r in recs) / len(recs)
        log(f"    stride={s:<4}: KLA−Fair átlag = {avg:+.2f}%  "
            f"({'KLA jobb' if avg < 0 else 'Fair jobb'})")

elapsed = time.time() - t_sweep_start
log(f"\n  Teljes futási idő: {elapsed/60:.1f} perc")
log("=" * 100)

# ─────────────────────────────────────────────────────────────────────────────
# Kiírás fájlba
# ─────────────────────────────────────────────────────────────────────────────

TXT_FILE.write_text("\n".join(summary_lines), encoding="utf-8")
print(f"\n  Szöveges riport: {TXT_FILE}")

# JSON (sample nélkül, hogy ne legyen gigantikus)
for r in all_results:
    for v in r.get("results", {}).values():
        v.pop("sample", None)

JSON_FILE.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"  JSON adatok   : {JSON_FILE}")
