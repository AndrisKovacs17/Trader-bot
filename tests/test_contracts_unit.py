#!/usr/bin/env python3
"""Unit tests for core/ops/contracts.py — 58 % → target ≥ 95 %."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ops.contracts import HealthStatus, SimpleMetrics


# ── HealthStatus ──────────────────────────────────────────────────────────────

def test_health_status_ok() -> None:
    h = HealthStatus(ok=True)
    assert h.ok is True
    assert h.details == {}


def test_health_status_not_ok_with_details() -> None:
    h = HealthStatus(ok=False, details={"error": "connection refused"})
    assert h.ok is False
    assert h.details["error"] == "connection refused"


def test_health_status_details_default_is_new_dict_per_instance() -> None:
    h1 = HealthStatus(ok=True)
    h2 = HealthStatus(ok=True)
    h1.details["x"] = 1
    assert "x" not in h2.details


# ── SimpleMetrics ─────────────────────────────────────────────────────────────

def test_simple_metrics_counter_increments() -> None:
    m = SimpleMetrics()
    m.counter("requests", {"endpoint": "/api"})
    m.counter("requests", {"endpoint": "/api"})
    key = "counter:requests:{'endpoint': '/api'}"
    assert m.values[key] == 2.0


def test_simple_metrics_counter_different_labels_independent() -> None:
    m = SimpleMetrics()
    m.counter("hits", {"route": "/"})
    m.counter("hits", {"route": "/health"})
    assert m.values["counter:hits:{'route': '/'}"] == 1.0
    assert m.values["counter:hits:{'route': '/health'}"] == 1.0


def test_simple_metrics_gauge_initialises_to_zero() -> None:
    m = SimpleMetrics()
    val = m.gauge("cpu", {"host": "node1"})
    assert val == 0.0


def test_simple_metrics_gauge_key_persists() -> None:
    m = SimpleMetrics()
    m.gauge("mem", {})
    key = "gauge:mem:{}"
    assert key in m.values


def test_simple_metrics_histogram_initialises_to_zero() -> None:
    m = SimpleMetrics()
    val = m.histogram("latency_ms", {"op": "predict"})
    assert val == 0.0


def test_simple_metrics_histogram_key_persists() -> None:
    m = SimpleMetrics()
    m.histogram("size", {"bucket": "large"})
    key = "hist:size:{'bucket': 'large'}"
    assert key in m.values


def test_simple_metrics_isolated_between_instances() -> None:
    m1 = SimpleMetrics()
    m2 = SimpleMetrics()
    m1.counter("x", {})
    assert m2.values == {}


def test_simple_metrics_counter_returns_incremented_value() -> None:
    m = SimpleMetrics()
    ret = m.counter("n", {})
    assert ret == 1.0
    ret2 = m.counter("n", {})
    assert ret2 == 2.0
