from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import logging
import math
import re
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import feedparser
import pandas as pd

logger = logging.getLogger(__name__)


# RSS források listája a kérésed alapján.
RSS_URLS = [
    "https://finance.yahoo.com/rss/headline?s=BTC-USD",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",
]

NEWS_COLUMNS = ["Time", "Title", "Source", "EntryId", "FetchedAt"]


@dataclass(slots=True)
class SentimentConfig:
    positive_threshold: float = 0.05
    negative_threshold: float = -0.05
    model_weight: float = 0.7
    lexicon_weight: float = 0.3

    def __post_init__(self) -> None:
        if not (-1.0 <= self.negative_threshold <= 1.0 and -1.0 <= self.positive_threshold <= 1.0):
            raise ValueError("Sentiment thresholds must be in [-1, 1].")
        if self.model_weight < 0 or self.lexicon_weight < 0:
            raise ValueError("Sentiment weights must be non-negative.")
        if self.model_weight + self.lexicon_weight == 0:
            raise ValueError("At least one sentiment weight must be > 0.")


CRYPTO_FINANCE_LEXICON: dict[str, float] = {
    "bullish": 0.55,
    "breakout": 0.45,
    "surge": 0.35,
    "rally": 0.30,
    "etf": 0.20,
    "upgrade": 0.25,
    "partnership": 0.20,
    "adoption": 0.25,
    "approval": 0.25,
    "bearish": -0.55,
    "crackdown": -0.70,
    "ban": -0.60,
    "lawsuit": -0.55,
    "hack": -0.70,
    "exploit": -0.70,
    "liquidation": -0.45,
    "default": -0.45,
    "insolvency": -0.60,
    "fraud": -0.70,
    "downgrade": -0.35,
    "outflow": -0.30,
}


class _LexiconOnlyAnalyzer:
    def polarity_scores(self, text: str) -> dict[str, float]:
        return {"compound": _domain_lexicon_score(text)}


def _to_utc_timestamp(value: Any) -> datetime | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _domain_lexicon_score(text: str) -> float:
    tokens = re.findall(r"[a-zA-Z]+", (text or "").lower())
    if not tokens:
        return 0.0
    score = 0.0
    hits = 0
    for token in tokens:
        if token in CRYPTO_FINANCE_LEXICON:
            score += CRYPTO_FINANCE_LEXICON[token]
            hits += 1
    if hits == 0:
        return 0.0
    return max(-1.0, min(1.0, score / max(1, hits)))


def _blend_sentiment_scores(model_score: float, lexicon_score: float, cfg: SentimentConfig) -> float:
    total_weight = cfg.model_weight + cfg.lexicon_weight
    blended = ((cfg.model_weight * model_score) + (cfg.lexicon_weight * lexicon_score)) / total_weight
    return max(-1.0, min(1.0, blended))


# Egyetlen RSS URL feldolgozása.
def fetch_rss(url: str) -> list[dict[str, Any]]:
    # Timeouttal töltjük le a feedet, hogy hálózati hiba esetén ne akadjon meg a pipeline.
    try:
        request = Request(url, headers={"User-Agent": "diplomamunkakod-news-adapter/1.0"})
        with urlopen(request, timeout=10) as response:
            raw = response.read()
        feed = feedparser.parse(raw)
    except Exception as error:
        # RSS rate limit vagy hálózati hiba esetén üres listát adunk vissza.
        logger.warning("RSS fetch failed for %s: %s", url, error)
        return []

    output: list[dict[str, Any]] = []
    fetched_at = datetime.now(timezone.utc)
    source_host = urlparse(url).netloc or "unknown"
    for entry in feed.entries:
        published = entry.get("published") or entry.get("updated") or entry.get("created")
        event_time = _to_utc_timestamp(published)
        if event_time is None:
            published_parsed = entry.get("published_parsed") or entry.get("updated_parsed")
            if published_parsed is not None:
                event_time = datetime(*published_parsed[:6], tzinfo=timezone.utc)
        if event_time is None:
            continue
        title = str(entry.get("title") or "").strip()
        if not title:
            continue
        row_fingerprint = f"{source_host}|{event_time.isoformat()}|{title or ''}"
        entry_id = hashlib.sha1(row_fingerprint.encode("utf-8")).hexdigest()[:16]
        output.append(
            {
                "Time": event_time,
                "Title": title,
                "Source": source_host,
                "EntryId": entry_id,
                "FetchedAt": fetched_at,
            }
        )
    return output


# Több RSS forrás párhuzamos letöltése, tisztítás, rendezés.
def get_all_news(rss_urls: list[str] | None = None) -> pd.DataFrame:
    urls = rss_urls or RSS_URLS

    with ThreadPoolExecutor(max_workers=max(1, len(urls))) as executor:
        results = list(executor.map(fetch_rss, urls))

    flat_list = [item for sublist in results for item in sublist]
    df = pd.DataFrame(flat_list)

    if df.empty:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    # Hibás dátumok NaT-ra, majd eldobás.
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"])
    if "FetchedAt" in df.columns:
        df["FetchedAt"] = pd.to_datetime(df["FetchedAt"], errors="coerce", utc=True)

    # Timezone egységesítés UTC-aware formára.
    if df["Time"].dt.tz is None:
        df["Time"] = df["Time"].dt.tz_localize("UTC")
    else:
        df["Time"] = df["Time"].dt.tz_convert("UTC")

    # Dedup + deterministic ordering (RSS ordering is not guaranteed).
    subset = [column for column in ["EntryId", "Source", "Title", "Time"] if column in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="first")
    sort_columns = [column for column in ["Time", "FetchedAt", "Source", "Title"] if column in df.columns]
    sort_order = [False if column in {"Time", "FetchedAt"} else True for column in sort_columns]
    return df.sort_values(sort_columns, ascending=sort_order).reset_index(drop=True)


@lru_cache(maxsize=1)
def _get_sentiment_analyzer() -> Any:
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.data.find("sentiment/vader_lexicon")
        return SentimentIntensityAnalyzer()
    except Exception:
        logger.warning("VADER lexicon unavailable locally; using lexicon-only fallback analyzer.")
        return _LexiconOnlyAnalyzer()


# VADER sentiment oszlopok hozzáadása.
def add_sentiment_columns(news_df: pd.DataFrame, config: SentimentConfig | None = None) -> pd.DataFrame:
    cfg = config or SentimentConfig()
    if news_df.empty:
        out = news_df.copy()
        out["Model_Sentiment_Score"] = pd.Series(dtype="float64")
        out["Lexicon_Sentiment_Score"] = pd.Series(dtype="float64")
        out["Sentiment_Score"] = pd.Series(dtype="float64")
        out["Sentiment"] = pd.Series(dtype="object")
        return out

    sia = _get_sentiment_analyzer()

    out = news_df.copy()
    out["Model_Sentiment_Score"] = out["Title"].fillna("").apply(lambda text: float(sia.polarity_scores(text)["compound"]))
    out["Lexicon_Sentiment_Score"] = out["Title"].fillna("").apply(_domain_lexicon_score)
    total_weight = cfg.model_weight + cfg.lexicon_weight
    out["Sentiment_Score"] = (
        (cfg.model_weight * out["Model_Sentiment_Score"])
        + (cfg.lexicon_weight * out["Lexicon_Sentiment_Score"])
    ) / total_weight
    out["Sentiment_Score"] = out["Sentiment_Score"].clip(-1.0, 1.0)
    out["Sentiment"] = out["Sentiment_Score"].apply(
        lambda value: "Positive"
        if value > cfg.positive_threshold
        else ("Negative" if value < cfg.negative_threshold else "Neutral")
    )
    return out


def add_rolling_sentiment_features(
    news_df: pd.DataFrame,
    window_minutes: int = 60,
    half_life_minutes: float = 30.0,
) -> pd.DataFrame:
    """Add time-windowed sentiment features suitable for trading models."""
    if news_df.empty:
        out = news_df.copy()
        out["Sentiment_Rolling_Mean"] = pd.Series(dtype="float64")
        out["Sentiment_Decay_Weighted"] = pd.Series(dtype="float64")
        out["News_Volume_Window"] = pd.Series(dtype="int64")
        return out

    out = news_df.copy()
    if "Time" not in out.columns:
        out["Sentiment_Rolling_Mean"] = pd.Series(dtype="float64")
        out["Sentiment_Decay_Weighted"] = pd.Series(dtype="float64")
        out["News_Volume_Window"] = pd.Series(dtype="int64")
        return out

    if "Sentiment_Score" not in out.columns:
        out = add_sentiment_columns(out)

    out["Time"] = pd.to_datetime(out["Time"], errors="coerce", utc=True)
    out = out.dropna(subset=["Time"]).sort_values("Time", ascending=True).reset_index(drop=True)

    window = pd.Timedelta(minutes=max(1, int(window_minutes)))
    half_life = max(1e-6, float(half_life_minutes))
    decay_lambda = math.log(2.0) / half_life

    rolling_mean: list[float] = []
    decay_weighted: list[float] = []
    volume_window: list[int] = []

    time_values: list[datetime] = [ts.to_pydatetime() for ts in out["Time"].tolist()]
    score_values: list[float] = out["Sentiment_Score"].fillna(0.0).astype(float).tolist()

    active_items: deque[tuple[datetime, float]] = deque()

    for now, score in zip(time_values, score_values, strict=False):
        window_start = now - window
        while active_items and active_items[0][0] < window_start:
            active_items.popleft()

        active_items.append((now, score))

        current_count = len(active_items)
        volume_window.append(current_count)

        if current_count == 0:
            rolling_mean.append(0.0)
            decay_weighted.append(0.0)
            continue

        window_sum = sum(item_score for _, item_score in active_items)
        rolling_mean.append(float(window_sum / current_count))

        weighted_sum = 0.0
        weight_sum = 0.0
        for item_time, item_score in active_items:
            age_minutes = max(0.0, (now - item_time).total_seconds() / 60.0)
            weight = math.exp(-decay_lambda * age_minutes)
            weighted_sum += item_score * weight
            weight_sum += weight

        decay_weighted.append(float(weighted_sum / weight_sum) if weight_sum > 0 else 0.0)

    out["Sentiment_Rolling_Mean"] = rolling_mean
    out["Sentiment_Decay_Weighted"] = decay_weighted
    out["News_Volume_Window"] = volume_window
    return out


# Opcionális stílus helper (notebook/display esethez).
def color_sentiment(value: object) -> str:
    if isinstance(value, float):
        color = "red" if value < -0.05 else "green" if value > 0.05 else "black"
    else:
        color = "red" if value == "Negative" else "green" if value == "Positive" else "black"
    return f"color: {color}"
