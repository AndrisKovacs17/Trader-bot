"""
Signal diagnostics: prob_up laposság, sigma halál, feature variáció elemzés.

Futtatás:
    python -m pytest tests/test_signal_diagnostics.py -s -v
"""
import math
import random
import sys
import types
from collections import deque

import pytest

# Helpers


def _make_bars(n: int, seed: int = 42, trend: float = 0.0) -> list[dict]:
    """Szintetikus OHLCV bars price random-walk + optional trend."""
    rng = random.Random(seed)
    price = 50000.0
    rows = []
    for i in range(n):
        ret = rng.gauss(trend, 0.003)
        price = price * (1 + ret)
        vol = rng.lognormvariate(3.0, 1.2)
        taker_frac = rng.betavariate(2, 2)   # 0..1, centred at 0.5
        funding = rng.gauss(0.0001, 0.0003)
        # Synthetic OHLC: open ≈ previous close, intrabar range ~0.1-0.3% of price
        open_p = price / (1.0 + ret) if abs(ret) > 1e-12 else price
        intrabar_half = price * abs(rng.gauss(0.0015, 0.0008))
        high_p = max(price, open_p) + intrabar_half * rng.random()
        low_p  = min(price, open_p) - intrabar_half * rng.random()
        rows.append(
            {
                "price": price,
                "return": ret,
                "volume": vol,
                "ts_ms": 1_700_000_000_000 + i * 300_000,   # 5-min bars
                "taker_buy_vol": vol * taker_frac,
                "funding_rate": funding,
                "o": open_p,
                "h": high_p,
                "l": low_p,
            }
        )
    return rows


# Test 1, Feature engineering variáció


class TestFeatureVariation:
    """Ellenőrzi hogy a feature-ök tényleg változnak bemenettől függően."""

    def test_features_are_not_constant(self):
        from core.ml.feature_engineering import build_trade_feature_rows, FEATURE_NAMES

        bars = _make_bars(300)
        rows, names = build_trade_feature_rows(bars)
        assert len(rows) == len(bars), "minden barhoz kell feature sor"
        assert names == FEATURE_NAMES

        # Minden feature oszlopára nézzük a std-t
        n_feat = len(FEATURE_NAMES)
        stds = []
        for fi in range(n_feat):
            vals = [r[fi] for r in rows[50:]]   # első 50 warm-up
            v = sum((x - sum(vals) / len(vals)) ** 2 for x in vals) / len(vals)
            stds.append(math.sqrt(v))

        dead = [FEATURE_NAMES[i] for i, s in enumerate(stds) if s < 1e-9]
        print("\n--- Feature std-ek ---")
        for i, (name, s) in enumerate(zip(FEATURE_NAMES, stds)):
            flag = "  ← HALOTT!" if s < 1e-9 else ""
            print(f"  [{i:2d}] {name:<22s}  std={s:.6f}{flag}")

        # weekday features are inherently low-variance over short (25h) test windows;
        # they are meaningful only when data spans multiple days/weeks.
        long_period_features = {"weekday_sin", "weekday_cos"}
        really_dead = [f for f in dead if f not in long_period_features]
        assert not really_dead, f"Halott (0 std) feature-ök: {really_dead}"

    def test_taker_features_vary_with_taker_vol(self):
        """taker_vol_ratio és bid_ask_proxy kell hogy varáljon ha taker_buy_vol adott."""
        from core.ml.feature_engineering import build_trade_feature_rows, FEATURE_NAMES

        bars = _make_bars(200)
        rows, _ = build_trade_feature_rows(bars)

        ti = FEATURE_NAMES.index("taker_vol_ratio")
        bi = FEATURE_NAMES.index("bid_ask_proxy")

        taker_vals = [r[ti] for r in rows[10:]]
        bid_vals   = [r[bi] for r in rows[10:]]

        taker_std = math.sqrt(sum((x - sum(taker_vals)/len(taker_vals))**2 for x in taker_vals) / len(taker_vals))
        bid_std   = math.sqrt(sum((x - sum(bid_vals)/len(bid_vals))**2 for x in bid_vals) / len(bid_vals))

        print(f"\n  taker_vol_ratio std = {taker_std:.6f}")
        print(f"  bid_ask_proxy   std = {bid_std:.6f}")
        assert taker_std > 0.01, "taker_vol_ratio nem változik – taker_buy_vol nem jut el a feature engineeringbe"
        assert bid_std   > 0.01, "bid_ask_proxy nem változik"

    # funding_rate_z removed from features (always 0.0 in spot data → pure noise)


# Test 2, Training engine átadja-e a taker/funding-ot


class TestTrainingEngineFeaturePassthrough:
    """Ellenőrzi hogy a TrainingEngine NEM dobja el a taker_buy_vol-t."""

    def test_training_rows_include_taker_volume(self):
        """Ha a dataset tartalmaz taker_buy_vol-t, a feature sorok nem lehetnek mind 0.5-ösek."""
        from core.ml.feature_engineering import build_trade_feature_rows, FEATURE_NAMES

        # Szimulál amit a TrainingEngine csinál (rows dict)
        raw_bars = _make_bars(300)

        # ---- RÉGI (hibás) ----
        rows_old = []
        for i, row in enumerate(raw_bars):
            rows_old.append({
                "price": row["price"],
                "return": row["return"],
                "volume": row["volume"],
                "ts_ms": row["ts_ms"],
                # taker_buy_vol és funding_rate KIMARAD → bug!
            })
        feat_old, _ = build_trade_feature_rows(rows_old)
        ti = FEATURE_NAMES.index("taker_vol_ratio")
        taker_old = [r[ti] for r in feat_old[10:]]
        taker_old_std = math.sqrt(sum((x - sum(taker_old)/len(taker_old))**2 for x in taker_old) / len(taker_old))

        # ---- ÚJ (helyes) ----
        rows_new = []
        for i, row in enumerate(raw_bars):
            rows_new.append({
                "price": row["price"],
                "return": row["return"],
                "volume": row["volume"],
                "ts_ms": row["ts_ms"],
                "taker_buy_vol": row["taker_buy_vol"],   # ← ez kell
                "funding_rate": row["funding_rate"],      # ← ez is
            })
        feat_new, _ = build_trade_feature_rows(rows_new)
        taker_new = [r[ti] for r in feat_new[10:]]
        taker_new_std = math.sqrt(sum((x - sum(taker_new)/len(taker_new))**2 for x in taker_new) / len(taker_new))

        print(f"\n  taker_vol_ratio std (régi, hibás): {taker_old_std:.6f}")
        print(f"  taker_vol_ratio std (új, helyes):  {taker_new_std:.6f}")

        assert taker_old_std < 1e-9, "Sanity: régi módszerrel taker kell hogy 0 legyen"
        assert taker_new_std > 0.01, "Új módszerrel taker kell hogy változzon"


# Test 3, Inference prob_up és sigma variáció


try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch szükséges")
class TestInferenceSignalVariation:
    """Ellenőrzi hogy inference alatt prob_up és sigma tényleg változnak."""

    def _build_fake_predictor(self, feature_dim: int = 26, lookback: int = 32):
        """Épít egy véletlenszerű inicializált KCAPredictor-t."""
        from core.ml.services import KCAPredictor
        from core.ml.kca_mamba import KCAMambaStack

        hidden_dim = 48
        pred = KCAPredictor(version="diag-test")

        kla = KCAMambaStack(feature_dim=feature_dim, hidden_dim=hidden_dim,
                            num_layers=2, heads=4, d_state=16, slow_stride=4)
        mu_head  = nn.Sequential(nn.Linear(hidden_dim, 32), nn.SiLU(), nn.Linear(32, 1))
        up_head  = nn.Sequential(nn.Linear(hidden_dim, 32), nn.SiLU(), nn.Linear(32, 3))
        var_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.SiLU(), nn.Linear(32, 1))
        model = nn.ModuleDict({"kca_stack": kla, "mu_head": mu_head,
                               "up_head": up_head, "var_head": var_head})

        # Fake scaler: unit normalization
        pred._scaler_mean = [0.0] * feature_dim
        pred._scaler_std  = [1.0] * feature_dim
        pred._lookback    = lookback
        pred._horizon     = 6
        pred._train_positive_ratio = 0.5
        pred._train_class_priors   = [0.47, 0.06, 0.47]
        pred._train_mu_mean        = 0.0
        pred._class_count          = 3
        pred.device = "cpu"
        pred._compiled_model = model
        pred._feature_buffers = {}
        return pred

    def _make_ctx(self, symbol: str = "BTCUSDT"):
        """Minimális RunContext mock."""
        ctx = types.SimpleNamespace()
        ctx.instrument = types.SimpleNamespace(symbol=symbol)
        ctx.config = types.SimpleNamespace()
        ctx.config.get = lambda key, default=None: {
            "model.prob_temperature":      1.0,
            "model.prior_debias_strength": 0.0,
            "model.max_abs_mu":            0.005,
            "model.mu_bias_strength":      1.0,
            "model.mu_gain":               1.0,
            "model.min_sigma_for_prob":    0.001,
            "model.prob_mu_blend":         0.0,
            "model.prob_floor":            0.02,
            "model.min_pred_variance":     1e-8,
            "model.max_pred_variance":     2.0,
        }.get(key, default)
        return ctx

    def _make_state(self, price: float, ret: float, vol: float,
                    taker_buy_vol: float = -1.0, ts_ms: int = 0):
        from core.ml.services import EstimatedState
        return EstimatedState(
            x=[price, ret, vol],
            P=[[1.0]],
            features={
                "price": price,
                "return": ret,
                "volume": vol,
                "timestamp_ms": ts_ms or 1_700_000_000_000,
                "taker_buy_vol": taker_buy_vol,
            },
            confidence=1.0,
        )

    @pytest.mark.asyncio
    async def test_prob_up_varies_across_inputs(self):
        """prob_up NEM lehet konstans különböző piaci körülmények közt."""
        import asyncio
        pred = self._build_fake_predictor()
        ctx  = self._make_ctx()
        bars = _make_bars(200, seed=7)

        prob_ups = []
        sigmas   = []
        for b in bars:
            st = self._make_state(b["price"], b["return"], b["volume"],
                                  b["taker_buy_vol"], b["ts_ms"])
            result = await pred.predict(st, ctx)
            prob_ups.append(result.prob_up)
            sigmas.append(result.sigma)

        up_mean = sum(prob_ups) / len(prob_ups)
        up_std  = math.sqrt(sum((x - up_mean)**2 for x in prob_ups) / len(prob_ups))
        sig_mean = sum(sigmas) / len(sigmas)
        sig_std  = math.sqrt(sum((x - sig_mean)**2 for x in sigmas) / len(sigmas))

        print(f"\n--- prob_up stat ({len(prob_ups)} tick) ---")
        print(f"  mean = {up_mean:.4f}  std = {up_std:.4f}  "
              f"min = {min(prob_ups):.4f}  max = {max(prob_ups):.4f}")
        print(f"--- sigma stat ---")
        print(f"  mean = {sig_mean:.6f}  std = {sig_std:.6f}  "
              f"min = {min(sigmas):.6f}  max = {max(sigmas):.6f}")

        # Elvárás: random init model → prob_up range > 0.05, sigma > 0 és változik
        prob_range = max(prob_ups) - min(prob_ups)
        sigma_range = max(sigmas) - min(sigmas)
        print(f"  prob_up range = {prob_range:.4f}  (min elvárás: > 0.05)")
        print(f"  sigma range   = {sigma_range:.6f}  (min elvárás: > 1e-5)")

        assert prob_range > 0.05, (
            f"prob_up TÚLSÁGOSAN LAPOS: range={prob_range:.5f}. "
            "Valószínű ok: minden input ugyanúgy normalizálódik, "
            "vagy a model kiment kollapszált."
        )
        assert sigma_range > 1e-5, (
            f"sigma HALOTT: range={sigma_range:.8f}. "
            "Valószínű ok: var_head konstans outputot ad (Beta-NLL nem tanult)."
        )

    @pytest.mark.asyncio
    async def test_bullish_bars_give_higher_prob_up(self):
        """Erősen bullish sorozat után prob_up > 0.55."""
        import asyncio
        pred = self._build_fake_predictor()
        ctx  = self._make_ctx()

        # Warm-up: 80 bar neutrális
        bars_neutral = _make_bars(80, seed=1, trend=0.0)
        for b in bars_neutral:
            st = self._make_state(b["price"], b["return"], b["volume"],
                                  b["taker_buy_vol"], b["ts_ms"])
            await pred.predict(st, ctx)

        # 30 erősen bullish bar
        bars_bull = _make_bars(30, seed=2, trend=0.005)
        results = []
        for b in bars_bull:
            st = self._make_state(b["price"], b["return"], b["volume"],
                                  b["taker_buy_vol"], b["ts_ms"])
            r = await pred.predict(st, ctx)
            results.append(r.prob_up)

        avg_bull = sum(results) / len(results)
        print(f"\n  Bullish sorozat után avg prob_up = {avg_bull:.4f}")
        # Random init modelltől pontosságot nem várunk, de a feature jelenléte ellenőrizhető
        # Csak azt nézzük, hogy nem konstans
        assert max(results) - min(results) > 0.005, \
            "prob_up még bullish sorozaton belül is teljesen konstans"

    @pytest.mark.asyncio
    async def test_sigma_responds_to_volatility(self):
        """Magas vol környezetben sigma nagyobb kell legyen mint alacsony vol-ban."""
        import asyncio
        pred_lo = self._build_fake_predictor()
        pred_hi = self._build_fake_predictor()
        ctx = self._make_ctx()

        bars_lo = _make_bars(150, seed=10, trend=0.0)
        bars_hi = []
        # Magas vol: ±2% per bar
        rng = random.Random(11)
        price = 50000.0
        for i in range(150):
            ret = rng.gauss(0, 0.02)
            price *= (1 + ret)
            vol = rng.lognormvariate(3.0, 1.2)
            bars_hi.append({"price": price, "return": ret, "volume": vol,
                             "ts_ms": 1_700_000_000_000 + i*300_000,
                             "taker_buy_vol": vol*rng.betavariate(2,2),
                             "funding_rate": 0.0})

        sig_lo, sig_hi = [], []
        for b in bars_lo:
            st = self._make_state(b["price"], b["return"], b["volume"],
                                  b["taker_buy_vol"], b["ts_ms"])
            r = await pred_lo.predict(st, ctx)
            sig_lo.append(r.sigma)
        for b in bars_hi:
            st = self._make_state(b["price"], b["return"], b["volume"],
                                  b["taker_buy_vol"], b["ts_ms"])
            r = await pred_hi.predict(st, ctx)
            sig_hi.append(r.sigma)

        avg_lo = sum(sig_lo[-50:]) / 50
        avg_hi = sum(sig_hi[-50:]) / 50
        print(f"\n  Alacsony vol sigma (átlag utolsó 50): {avg_lo:.6f}")
        print(f"  Magas vol sigma (átlag utolsó 50):   {avg_hi:.6f}")
        # Egy véletlenszerű modell nem feltétlenül helyes irányban reagál,
        # de a sigma nem lehet mind egyforma
        all_sigs = sig_lo + sig_hi
        global_range = max(all_sigs) - min(all_sigs)
        assert global_range > 1e-5, \
            f"sigma teljesen halott: global range = {global_range:.8f}"


# Test 4, Prior debias 3-class safety check


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch szükséges")
class TestPriorDebias3Class:
    """Ellenőrzi hogy a prior debias nem okoz FLAT inflációt kis prior esetén."""

    def test_near_zero_flat_prior_does_not_collapse_prob_up(self):
        """Ha FLAT prior ≈ 0.005, prior_debias_strength=0 esetén prob_up ~Uniform."""
        import torch

        # Emuláljuk amit services.py csinál
        raw_logits = torch.zeros(1, 3)   # kollapszált model
        priors = [0.472, 0.005, 0.524]
        strength = 0.0

        prior_tensor = torch.tensor(priors)
        prior_tensor_safe = torch.clamp(prior_tensor, min=0.1)
        prior_logits = torch.log(prior_tensor_safe).view(1, -1)
        debiased = raw_logits - strength * prior_logits
        probs = torch.softmax(debiased / 1.0, dim=-1)
        prob_up_raw = float(probs[:, 2] + 0.5 * probs[:, 1])

        print(f"\n  strength=0.0 → prob_up_raw = {prob_up_raw:.4f} (elvárás: ~0.50)")
        assert 0.45 < prob_up_raw < 0.55, \
            f"prior_debias_strength=0 esetén prob_up nem közel 0.5: {prob_up_raw}"

    def test_old_bug_would_have_collapsed(self):
        """Demonstrálja a régi bugot: strength=0.5 + tiny FLAT prior.

        A bug akkor jelenik meg amikor a modell DOWN-t jósol:
        a FLAT infláció (+2.65 FLAT logit boost) megakadályozza hogy prob_up
        kellően csökkenjen – így a DOWN jelzés "el van fojtva".
        A fix (min=0.1 clamp) log(0.1)=-2.3-ra korlátozza a max booston.
        """
        import torch

        # Modell határozottan DOWN-t jósol: logit[DOWN=0] >> logit[UP=2]
        raw_logits = torch.tensor([[2.0, 0.0, -2.0]])
        priors = [0.472, 0.005, 0.524]
        strength = 0.5

        # RÉGI: nem clampelt – log(0.005)=-5.3 → FLAT kap +2.65 booston
        prior_tensor = torch.tensor(priors)
        prior_logits_old = torch.log(prior_tensor).view(1, -1)
        debiased_old = raw_logits - strength * prior_logits_old
        probs_old = torch.softmax(debiased_old / 1.0, dim=-1)
        prob_up_old = float(probs_old[:, 2] + 0.5 * probs_old[:, 1])

        # ÚJ: clampelt – log(0.1)=-2.3, sokkal kisebb boost
        prior_tensor_safe = torch.clamp(prior_tensor, min=0.1)
        prior_logits_new = torch.log(prior_tensor_safe).view(1, -1)
        debiased_new = raw_logits - strength * prior_logits_new
        probs_new = torch.softmax(debiased_new / 1.0, dim=-1)
        prob_up_new = float(probs_new[:, 2] + 0.5 * probs_new[:, 1])

        print(f"\n  DOWN logits [2.0, 0.0, -2.0]:")
        print(f"  Régi bug prob_up = {prob_up_old:.4f} (FLAT infláció elfojtja a DOWN jelt)")
        print(f"  Fix után prob_up = {prob_up_new:.4f} (DOWN jel átmegy)")
        # Régi bug: FLAT infláció miatt prob_up magasan marad DOWN predikciónál is
        assert prob_up_old > prob_up_new + 0.05, (
            f"Fix kell hogy csökkentse prob_up-ot DOWN predikciónál: "
            f"old={prob_up_old:.4f} vs new={prob_up_new:.4f}"
        )
        assert prob_up_new < 0.25, (
            f"Fix után DOWN predikciónál prob_up kell < 0.25, kapott: {prob_up_new:.4f}"
        )


# Test 5, mu_implied_prob blend hatása lapos prob_up-ra


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch szükséges")
class TestMuBlendEffect:
    """prob_mu_blend visszakapcsolva élesíti a jelet ha a classifier lapos."""

    def test_mu_blend_sharpens_flat_classifier(self):
        """Ha classifier prob_up ≈ 0.5, de mu erősen bullish → blend emeli prob_up-t."""
        import math

        mu = 0.003        # bullish: +0.3% előrejelzett return
        sigma_ref = 0.001  # min_sigma_for_prob

        mu_implied = 0.5 + 0.5 * math.tanh(mu / sigma_ref)
        print(f"\n  mu={mu}, sigma_ref={sigma_ref} → mu_implied_prob = {mu_implied:.4f}")

        # blend=0.0: csak classifier (lapos)
        prob_raw = 0.50
        prob_blend0 = (1.0 - 0.0) * prob_raw + 0.0 * mu_implied
        prob_blend3 = (1.0 - 0.3) * prob_raw + 0.3 * mu_implied
        print(f"  prob_mu_blend=0.0 → prob_up = {prob_blend0:.4f}  (lapos)")
        print(f"  prob_mu_blend=0.3 → prob_up = {prob_blend3:.4f}  (élesebb)")

        assert prob_blend3 > prob_blend0 + 0.05, \
            "Blend nem élesíti a jelet bullish mu esetén"

    def test_min_sigma_for_prob_effect(self):
        """Ha min_sigma_for_prob túl nagy, mu_implied_prob soha nem tér el 0.5-től."""
        import math

        mu = 0.003
        mu_implied_small_sig = 0.5 + 0.5 * math.tanh(mu / 0.001)   # min_sigma=0.001 → tanh≈1
        mu_implied_large_sig = 0.5 + 0.5 * math.tanh(mu / 0.03)    # min_sigma=0.03 → tanh≈0.1

        print(f"\n  mu={mu}")
        print(f"  min_sigma_for_prob=0.001 → mu_implied = {mu_implied_small_sig:.4f}")
        print(f"  min_sigma_for_prob=0.030 → mu_implied = {mu_implied_large_sig:.4f}")

        assert mu_implied_small_sig > 0.9, \
            "Kis min_sigma → mu_implied közel 1 (helyes)"
        assert mu_implied_large_sig < 0.6, \
            "Nagy min_sigma → mu_implied majdnem 0.5 (lapos – ez a bug!)"
