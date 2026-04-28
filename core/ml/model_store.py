"""Persisting and loading the active KCA model on disk.

Format on disk (under `models/`):
- `kca_active.bin`  : raw bytes produced by `OfflineTrainingEngine.export_weights()`
                     (8B big-endian version length || version utf-8 || torch.save payload)
- `kca_active.json` : metadata snapshot (val_acc, Brier, version, hyperparams subset)
                     used by the sweep tool and for human inspection.

The bin format is the same one consumed by `KCAPredictor.request_model_update()`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_MODEL_DIR = Path("models")
ACTIVE_BIN = "kca_active.bin"
ACTIVE_META = "kca_active.json"


def model_dir() -> Path:
    """Return the configured models directory (env override: KCA_MODEL_DIR)."""
    raw = os.getenv("KCA_MODEL_DIR", "").strip()
    return Path(raw) if raw else DEFAULT_MODEL_DIR


def active_paths() -> tuple[Path, Path]:
    base = model_dir()
    return base / ACTIVE_BIN, base / ACTIVE_META


def has_saved_active() -> bool:
    bin_path, _ = active_paths()
    return bin_path.is_file() and bin_path.stat().st_size > 16


def save_active(weights: bytes, *, version: str, metadata: dict[str, Any]) -> tuple[Path, Path]:
    """Atomically write the active model bytes + metadata sidecar.

    Returns (bin_path, meta_path).
    """
    base = model_dir()
    base.mkdir(parents=True, exist_ok=True)
    bin_path, meta_path = active_paths()

    tmp_bin = bin_path.with_suffix(bin_path.suffix + ".tmp")
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")

    tmp_bin.write_bytes(weights)
    payload = {"version": version, **metadata}
    tmp_meta.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    os.replace(tmp_bin, bin_path)
    os.replace(tmp_meta, meta_path)
    return bin_path, meta_path


def load_active() -> tuple[bytes, dict[str, Any]] | None:
    """Read saved bytes + metadata. Returns None if not present."""
    bin_path, meta_path = active_paths()
    if not bin_path.is_file():
        return None
    raw = bin_path.read_bytes()
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    return raw, meta


def parse_version(weights: bytes) -> str:
    """Best-effort: extract the version string from saved bytes."""
    if len(weights) < 8:
        return ""
    n = int.from_bytes(weights[:8], byteorder="big")
    if 8 + n > len(weights):
        return ""
    try:
        return weights[8:8 + n].decode("utf-8")
    except Exception:
        return ""
