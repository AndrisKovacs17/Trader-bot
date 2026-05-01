#!/usr/bin/env python3
"""Unit tests for core/ml/model_store.py — 0 % → target ≥ 90 %."""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import core.ml.model_store as ms


# ── helpers ──────────────────────────────────────────────────────────────────

def _encode(version: str, payload: bytes = b"data") -> bytes:
    encoded = version.encode("utf-8")
    return struct.pack(">Q", len(encoded)) + encoded + payload


# ── model_dir ────────────────────────────────────────────────────────────────

def test_model_dir_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KCA_MODEL_DIR", raising=False)
    assert ms.model_dir() == Path("models")


def test_model_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KCA_MODEL_DIR", str(tmp_path))
    assert ms.model_dir() == tmp_path


def test_model_dir_empty_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KCA_MODEL_DIR", "   ")
    assert ms.model_dir() == Path("models")


# ── active_paths ─────────────────────────────────────────────────────────────

def test_active_paths_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KCA_MODEL_DIR", str(tmp_path))
    bin_p, meta_p = ms.active_paths()
    assert bin_p.name == ms.ACTIVE_BIN
    assert meta_p.name == ms.ACTIVE_META
    assert bin_p.parent == tmp_path


# ── has_saved_active ──────────────────────────────────────────────────────────

def test_has_saved_active_false_when_no_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KCA_MODEL_DIR", str(tmp_path))
    assert ms.has_saved_active() is False


def test_has_saved_active_false_when_file_too_small(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KCA_MODEL_DIR", str(tmp_path))
    (tmp_path / ms.ACTIVE_BIN).write_bytes(b"\x00" * 10)
    assert ms.has_saved_active() is False


def test_has_saved_active_true_when_valid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KCA_MODEL_DIR", str(tmp_path))
    (tmp_path / ms.ACTIVE_BIN).write_bytes(b"\x00" * 32)
    assert ms.has_saved_active() is True


# ── save_active ───────────────────────────────────────────────────────────────

def test_save_active_creates_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KCA_MODEL_DIR", str(tmp_path))
    weights = _encode("v1", b"weights-payload")
    bin_p, meta_p = ms.save_active(weights, version="v1", metadata={"val_acc": 0.9})
    assert bin_p.read_bytes() == weights
    meta = json.loads(meta_p.read_text())
    assert meta["version"] == "v1"
    assert meta["val_acc"] == 0.9


def test_save_active_atomic_no_tmp_leftover(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KCA_MODEL_DIR", str(tmp_path))
    ms.save_active(b"\x00" * 32, version="v2", metadata={})
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == [], f"Temporary files left: {leftover}"


def test_save_active_overwrites_previous(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KCA_MODEL_DIR", str(tmp_path))
    ms.save_active(_encode("v1"), version="v1", metadata={"score": 1})
    ms.save_active(_encode("v2"), version="v2", metadata={"score": 2})
    _, meta_p = ms.active_paths()
    meta = json.loads(meta_p.read_text())
    assert meta["version"] == "v2"
    assert meta["score"] == 2


def test_save_active_creates_parent_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested"
    monkeypatch.setenv("KCA_MODEL_DIR", str(nested))
    ms.save_active(b"\x00" * 32, version="v0", metadata={})
    assert nested.exists()


# ── load_active ───────────────────────────────────────────────────────────────

def test_load_active_none_when_no_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KCA_MODEL_DIR", str(tmp_path))
    assert ms.load_active() is None


def test_load_active_returns_bytes_and_meta(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KCA_MODEL_DIR", str(tmp_path))
    weights = _encode("roundtrip")
    ms.save_active(weights, version="roundtrip", metadata={"loss": 0.05})
    result = ms.load_active()
    assert result is not None
    raw, meta = result
    assert raw == weights
    assert meta["version"] == "roundtrip"
    assert meta["loss"] == 0.05


def test_load_active_tolerates_missing_meta(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KCA_MODEL_DIR", str(tmp_path))
    (tmp_path / ms.ACTIVE_BIN).write_bytes(b"\x00" * 32)
    result = ms.load_active()
    assert result is not None
    raw, meta = result
    assert raw == b"\x00" * 32
    assert meta == {}


def test_load_active_tolerates_corrupt_meta(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KCA_MODEL_DIR", str(tmp_path))
    (tmp_path / ms.ACTIVE_BIN).write_bytes(b"\x00" * 32)
    (tmp_path / ms.ACTIVE_META).write_text("NOT JSON {{{{", encoding="utf-8")
    result = ms.load_active()
    assert result is not None
    _, meta = result
    assert meta == {}


# ── parse_version ─────────────────────────────────────────────────────────────

def test_parse_version_valid() -> None:
    data = _encode("my-model-v3")
    assert ms.parse_version(data) == "my-model-v3"


def test_parse_version_empty_string_when_too_short() -> None:
    assert ms.parse_version(b"\x00\x07") == ""


def test_parse_version_empty_string_when_length_exceeds_data() -> None:
    data = struct.pack(">Q", 999) + b"short"
    assert ms.parse_version(data) == ""


def test_parse_version_empty_bytes() -> None:
    assert ms.parse_version(b"") == ""


def test_parse_version_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KCA_MODEL_DIR", str(tmp_path))
    version = "sweep-001"
    weights = _encode(version, b"payload")
    ms.save_active(weights, version=version, metadata={})
    raw, _ = ms.load_active()
    assert ms.parse_version(raw) == version
