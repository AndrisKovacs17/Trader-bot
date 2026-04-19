from __future__ import annotations

from datetime import datetime
import math
from typing import Any

FEATURE_NAMES: list[str] = [
    # --- Core momentum / returns ---
    "ret",           # current-bar return
    "ret_mean_8",    # short-term drift (8-bar EMA of returns)
    "ret_vol_8",     # short-term volatility  ← individually helps on real BTC
    "ret_vol_32",    # long-term volatility   ← individually helps on real BTC
    # --- Order flow ---
    "taker_vol_ratio",   # taker buy / total volume → order-flow imbalance proxy
    "volume_accel",      # log(vol_now / vol_mean_8): volume acceleration
    "bid_ask_proxy",     # (2*taker_buy - total_vol) / total_vol, ≈ signed flow
    # --- Price structure ---
    "streak_8",
    "dist_ma_16",
    "dist_ma_64",
    "trend_ma_8_32",
    "ret_z_32",
    "breakout_32",
    "trend_quality_8",
    # --- Microstructure signals ---
    "vol_ret_align",  # vol_z_16 * ret / ret_vol_8 – vol-momentum alignment
    "ret_autocorr",   # ret * ret_lag1 / vol² – short-lag autocorrelation
    # --- Candle structure ---
    "body_ratio",    # |close-open| / (high-low) – directional conviction
    "upper_shadow",  # (high - max(open,close)) / range – rejection from top
    "lower_shadow",  # (min(open,close) - low) / range – rejection from bottom
    "engulfing",     # price vs prior bar close-open direction signal
    "dist_high_64",  # (64-bar high - price) / price – distance from resistance
    "dist_low_64",   # (price - 64-bar low) / price – distance from support
    # --- Technical oscillators ---
    "rsi_14",
    "bb_pos_20",
    "macd_norm",
    "stoch_k",
    # --- Time (cyclic) – biggest contributor on real BTC data ---
    "intraday_sin",  # sin(2π*hour/24) – captures Asia/EU/US session patterns
    "intraday_cos",  # cos(2π*hour/24) – pairs with sin for unambiguous encoding
    "weekday_sin",   # sin(2π*weekday/7) – day-of-week effect
    "weekday_cos",   # cos(2π*weekday/7) – pairs with sin
    # --- Volume ---
    "log_volume",
    # --- Extended trend context ---
    "ret_mean_32",   # 32-bar return EMA – medium-term momentum
    # --- Order flow (derived) ---
    "ofi_delta",      # taker_ratio – EMA8(taker_ratio) – order-flow acceleration
]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _to_ts_ms(value: Any, fallback_ms: int) -> int:
    if value is None:
        return fallback_ms
    if isinstance(value, datetime):
        return int(value.timestamp() * 1000)
    try:
        raw = int(value)
    except (TypeError, ValueError):
        return fallback_ms
    return max(raw, 0)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / float(n - 1)
    return math.sqrt(max(var, 0.0))


def _window(values: list[float], end_index: int, size: int) -> list[float]:
    start = max(0, end_index - size + 1)
    return values[start : end_index + 1]


def _ema(values: list[float], span: int) -> float:
    """Exponential moving average of *values* using smoothing factor 2/(span+1)."""
    if not values:
        return 0.0
    alpha = 2.0 / (span + 1)
    result = values[0]
    for v in values[1:]:
        result = alpha * v + (1.0 - alpha) * result
    return result


def build_trade_feature_rows(rows: list[dict[str, Any]]) -> tuple[list[list[float]], list[str]]:
    """Build deterministic engineered feature rows from sequential trade/bar rows.

    Expected input keys per row:
        price, return (optional), volume/qty (optional), timestamp/ts/ts_ms (optional),
        taker_buy_vol (optional, taker buy base volume for order-flow features),
        num_trades (optional, number of trades in bar),
        funding_rate (optional, perpetual funding rate for the bar).
    """
    if not rows:
        return [], list(FEATURE_NAMES)

    prices: list[float] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    volumes: list[float] = []
    returns: list[float] = []
    log_volumes: list[float] = []
    taker_ratios: list[float] = []  # taker_buy / total_vol
    funding_rates: list[float] = []
    ts_mss: list[int] = []
    feature_rows: list[list[float]] = []

    prev_price = 0.0
    prev_ts_ms = 0
    streak = 0

    for idx, row in enumerate(rows):
        price = _to_float(row.get("price", 0.0), default=prev_price if prev_price > 0 else 0.0)
        if price <= 0 and prev_price > 0:
            price = prev_price

        # OHLC – graceful fallback to close price when not available
        open_price = _to_float(row.get("open", row.get("o", price)), price)
        high_price = _to_float(row.get("high", row.get("h", price)), price)
        low_price  = _to_float(row.get("low",  row.get("l", price)), price)
        # Enforce consistency: high >= max(o, c), low <= min(o, c)
        high_price = max(high_price, price, open_price)
        low_price  = min(low_price,  price, open_price)

        raw_return = row.get("return", None)
        if raw_return is None:
            if prev_price > 0:
                ret = (price - prev_price) / prev_price
            else:
                ret = 0.0
        else:
            ret = _to_float(raw_return, 0.0)

        volume = _to_float(row.get("volume", row.get("qty", 0.0)), 0.0)
        log_volume = math.log1p(max(0.0, volume))

        # Taker buy volume (from kline field[9]); falls back to 0.5*total if absent
        taker_buy = _to_float(row.get("taker_buy_vol", -1.0), -1.0)
        if taker_buy < 0.0 or volume <= 0.0:
            taker_ratio = 0.5  # neutral when data unavailable
        else:
            taker_ratio = taker_buy / max(volume, 1e-8)
        taker_ratio = max(0.0, min(1.0, taker_ratio))

        # Funding rate (perpetual swap); 0.0 when unavailable (spot training data)
        funding_rate = _to_float(row.get("funding_rate", 0.0), 0.0)

        ts_value = row.get("ts_ms", row.get("timestamp", row.get("ts", None)))
        ts_ms = _to_ts_ms(ts_value, fallback_ms=prev_ts_ms + 1000 if prev_ts_ms > 0 else (idx + 1) * 1000)

        if ret > 0:
            streak = streak + 1 if streak >= 0 else 1
        elif ret < 0:
            streak = streak - 1 if streak <= 0 else -1
        else:
            streak = 0

        prices.append(price)
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        volumes.append(volume)
        returns.append(ret)
        log_volumes.append(log_volume)
        taker_ratios.append(taker_ratio)
        funding_rates.append(funding_rate)
        ts_mss.append(ts_ms)

        # ── volatility windows ──────────────────────────────────────────────
        ret_win_8 = _window(returns, idx, 8)
        ret_win_32 = _window(returns, idx, 32)
        ret_mean_8 = _mean(ret_win_8)
        ret_vol_8 = _std(ret_win_8)
        ret_vol_32 = _std(ret_win_32)

        vol_win_16 = _window(log_volumes, idx, 16)
        vol_mean_16 = _mean(vol_win_16)
        vol_std_16 = max(_std(vol_win_16), 1e-8)
        vol_z_16 = (log_volume - vol_mean_16) / vol_std_16

        # ── price structure ─────────────────────────────────────────────────
        ma_8 = _mean(_window(prices, idx, 8))
        ma_16 = _mean(_window(prices, idx, 16))
        ma_32 = _mean(_window(prices, idx, 32))
        ma_64 = _mean(_window(prices, idx, 64))

        dist_ma_16 = (price - ma_16) / max(abs(ma_16), 1e-8)
        dist_ma_64 = (price - ma_64) / max(abs(ma_64), 1e-8)
        trend_ma_8_32 = (ma_8 - ma_32) / max(abs(ma_32), 1e-8)

        streak_8 = max(-8.0, min(8.0, float(streak))) / 8.0

        # ── microstructure: order-flow ──────────────────────────────────────
        # taker_vol_ratio: raw taker buy fraction [0,1] → centred to [-0.5, 0.5]
        taker_vol_ratio = taker_ratio - 0.5  # positive = net buying pressure

        # volume_accel: log(current_vol / short_mean_vol) — positive = surge
        vol_mean_8 = math.exp(_mean(_window(log_volumes, idx, 8)))
        volume_accel = max(-3.0, min(3.0, math.log(max(volume, 1e-8) / max(vol_mean_8, 1e-8))))

        # bid_ask_proxy: (2*taker - total) / total = 2*taker_ratio - 1 ∈ [-1,1]
        # +1 = all taker buys (aggressive buyers), -1 = all taker sells
        bid_ask_proxy = max(-1.0, min(1.0, 2.0 * taker_ratio - 1.0))

        # ── funding rate z-score ────────────────────────────────────────────
        # z-score vs last 24 periods (8h * 24 = 8 days for 8h funding)
        _fr_win = _window(funding_rates, idx, 24)
        _fr_mean = _mean(_fr_win)
        _fr_std = max(_std(_fr_win), 1e-8)
        funding_rate_z = max(-3.0, min(3.0, (funding_rate - _fr_mean) / _fr_std))

        # ── RSI-14 ──────────────────────────────────────────────────────────
        _rsi_wins = _window(returns, idx, 14)
        if len(_rsi_wins) >= 2:
            _rsi_gain = sum(max(r, 0.0) for r in _rsi_wins) / len(_rsi_wins)
            _rsi_loss = sum(abs(min(r, 0.0)) for r in _rsi_wins) / len(_rsi_wins)
            if _rsi_loss < 1e-12:
                rsi_14 = 1.0 if _rsi_gain > 0 else 0.0
            else:
                _rs = _rsi_gain / _rsi_loss
                rsi_14 = (100.0 - 100.0 / (1.0 + _rs) - 50.0) / 50.0
        else:
            rsi_14 = 0.0

        # ── Bollinger Band %B ────────────────────────────────────────────────
        _pw20 = _window(prices, idx, 20)
        _ma20 = _mean(_pw20)
        _std20 = max(_std(_pw20), 1e-8)
        bb_pos_20 = max(-1.0, min(2.0, (price - (_ma20 - 2.0 * _std20)) / max(4.0 * _std20, 1e-8)))

        # ── ATR-14 z-score ──────────────────────────────────────────────────
        _atr_win = _window(returns, idx, 14)
        _atr_raw = sum(abs(r) for r in _atr_win) / max(len(_atr_win), 1)
        _atr_long = _window(returns, idx, 64)
        _atr_long_mean = sum(abs(r) for r in _atr_long) / max(len(_atr_long), 1)
        _atr_long_std = max(_std([abs(r) for r in _atr_long]), 1e-8)
        atr_14_z = max(-3.0, min(3.0, (_atr_raw - _atr_long_mean) / _atr_long_std))

        # ── Return z-score ──────────────────────────────────────────────────
        _ret_vol_32_ref = max(ret_vol_32, 1e-8)
        ret_z_32 = max(-4.0, min(4.0, ret / _ret_vol_32_ref))

        # ── Breakout / trend pattern ─────────────────────────────────────────
        _pw32 = _window(prices, idx, 32)
        _max32 = max(_pw32)
        _min32 = min(_pw32)
        _mid32 = (_max32 + _min32) / 2.0
        _half_range32 = max((_max32 - _min32) / 2.0, 1e-8)
        breakout_32 = max(-1.5, min(1.5, (price - _mid32) / _half_range32))

        _rw8 = _window(returns, idx, 8)
        _pos_bars = sum(1 for r in _rw8 if r > 1e-10)
        _neg_bars = sum(1 for r in _rw8 if r < -1e-10)
        trend_quality_8 = (_pos_bars - _neg_bars) / max(len(_rw8), 1)

        # ── MACD normalised ──────────────────────────────────────────────────
        _price_long = _window(prices, idx, 78)
        _ema12 = _ema(_price_long, 12)
        _ema26 = _ema(_price_long, 26)
        _macd_raw = (_ema12 - _ema26) / max(abs(price), 1e-8)
        macd_norm = max(-3.0, min(3.0, _macd_raw / _ret_vol_32_ref))

        # ── Stochastic %K ────────────────────────────────────────────────────
        _pw14 = _window(prices, idx, 14)
        _min14 = min(_pw14)
        _max14 = max(_pw14)
        stoch_k = max(-1.0, min(1.0, (price - _min14) / max(_max14 - _min14, 1e-8) * 2.0 - 1.0))

        # ── Volume-return alignment ──────────────────────────────────────────
        _ret_vol_8_ref = max(ret_vol_8, 1e-8)
        vol_ret_align = max(-3.0, min(3.0, vol_z_16 * ret / _ret_vol_8_ref))

        # ── Vol ratio & autocorr ─────────────────────────────────────────────
        vol_ratio_log = max(-3.0, min(3.0, math.log(max(ret_vol_8, 1e-8) / _ret_vol_32_ref)))
        _vol_sq = max(ret_vol_8 * _ret_vol_8_ref, 1e-12)
        ret_lag1 = returns[idx - 1] if idx >= 1 else 0.0
        ret_autocorr = max(-3.0, min(3.0, (ret * ret_lag1) / _vol_sq))

        # ── Cyclic time features ─────────────────────────────────────────────
        _frac_of_day = (ts_ms % 86_400_000) / 86_400_000.0
        intraday_sin = math.sin(2.0 * math.pi * _frac_of_day)
        intraday_cos = math.cos(2.0 * math.pi * _frac_of_day)
        # Weekday: Unix epoch (1970-01-01) was Thursday (Mon=0 → Thu=3)
        _weekday_frac = ((ts_ms // 86_400_000 + 3) % 7) / 7.0
        weekday_sin = math.sin(2.0 * math.pi * _weekday_frac)
        weekday_cos = math.cos(2.0 * math.pi * _weekday_frac)

        # ── Extended trend context ───────────────────────────────────────────
        ma_128 = _mean(_window(prices, idx, 128))
        dist_ma_128 = (price - ma_128) / max(abs(ma_128), 1e-8)
        ret_mean_32 = _mean(_window(returns, idx, 32))

        # ── Candlestick anatomy ──────────────────────────────────────────────
        _bar_range = max(high_price - low_price, 1e-8)
        body_ratio   = abs(price - open_price) / _bar_range
        upper_shadow = (high_price - max(open_price, price)) / _bar_range
        lower_shadow = (min(open_price, price) - low_price) / _bar_range
        # Engulfing: requires 2 bars
        if idx >= 1:
            _po, _pc = opens[idx - 1], prices[idx - 1]
            _prev_bull = _pc > _po
            _prev_bear = _pc < _po
            _curr_bull = price > open_price
            _curr_bear = price < open_price
            if _prev_bear and _curr_bull and open_price <= _pc and price >= _po:
                engulfing = 1.0   # bullish engulfing
            elif _prev_bull and _curr_bear and open_price >= _pc and price <= _po:
                engulfing = -1.0  # bearish engulfing
            else:
                engulfing = 0.0
        else:
            engulfing = 0.0

        # ── Support / resistance proximity (actual 64-bar H/L) ───────────────
        _h64 = _window(highs, idx, 64)
        _l64 = _window(lows,  idx, 64)
        _resistance_64 = max(_h64)
        _support_64    = min(_l64)
        dist_high_64 = (_resistance_64 - price) / max(price, 1e-8)
        dist_low_64  = (price - _support_64)    / max(price, 1e-8)

        # ── VWAP distance (64-bar) ───────────────────────────────────────────
        # VWAP = Σ(p_i * v_i) / Σ(v_i) over last 64 bars; volume-weighted anchor.
        # Positive = price above VWAP (bullish intraday bias), negative = below.
        _v64 = _window(volumes, idx, 64)
        _p64 = _window(prices, idx, 64)
        _vwap_denom = sum(_v64)
        if _vwap_denom > 1e-8:
            _vwap_64 = sum(p * v for p, v in zip(_p64, _v64)) / _vwap_denom
        else:
            _vwap_64 = _mean(_p64)
        vwap_dist_64 = max(-0.05, min(0.05, (price - _vwap_64) / max(price, 1e-8)))

        # ── Order Flow Imbalance delta ───────────────────────────────────────
        # Acceleration of taker buy pressure vs its own short EMA.
        # +ve = buy flow accelerating (momentum), -ve = fading.
        _ofi_ema8 = _ema(_window(taker_ratios, idx, 8), 8)
        ofi_delta = max(-1.0, min(1.0, taker_ratio - _ofi_ema8))

        # ── Hurst exponent via R/S (32-bar) ─────────────────────────────────
        # H > 0.5 → trending regime (momentum strategies profit)
        # H < 0.5 → mean-reverting regime (contrarian strategies profit)
        # Scaled to [-1, 1]: output = (H - 0.5) * 2
        _h_rets = _window(returns, idx, 32)
        _h_n = len(_h_rets)
        if _h_n >= 4:
            _h_mean = _mean(_h_rets)
            _h_std = max(_std(_h_rets), 1e-8)
            _h_cum = 0.0
            _h_cum_min = 0.0
            _h_cum_max = 0.0
            for _r in _h_rets:
                _h_cum += _r - _h_mean
                if _h_cum < _h_cum_min:
                    _h_cum_min = _h_cum
                if _h_cum > _h_cum_max:
                    _h_cum_max = _h_cum
            _h_R = _h_cum_max - _h_cum_min
            _h_hurst = math.log(max(_h_R / _h_std, 1e-8)) / math.log(_h_n)
            hurst_rs_32 = max(-1.0, min(1.0, (_h_hurst - 0.5) * 2.0))
        else:
            hurst_rs_32 = 0.0

        feature_row = [
            ret,
            ret_mean_8,
            ret_vol_8,
            ret_vol_32,
            taker_vol_ratio,
            volume_accel,
            bid_ask_proxy,
            streak_8,
            dist_ma_16,
            dist_ma_64,
            trend_ma_8_32,
            ret_z_32,
            breakout_32,
            trend_quality_8,
            vol_ret_align,
            ret_autocorr,
            body_ratio,
            upper_shadow,
            lower_shadow,
            engulfing,
            dist_high_64,
            dist_low_64,
            rsi_14,
            bb_pos_20,
            macd_norm,
            stoch_k,
            intraday_sin,
            intraday_cos,
            weekday_sin,
            weekday_cos,
            log_volume,
            ret_mean_32,
            ofi_delta,
        ]

        # Defensive finite clamp for model safety.
        safe_row = [v if math.isfinite(v) else 0.0 for v in feature_row]
        feature_rows.append(safe_row)

        prev_price = price
        prev_ts_ms = ts_ms

    return feature_rows, list(FEATURE_NAMES)
