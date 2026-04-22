from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import math
import os
import socket
import threading
from typing import Callable, Protocol
from urllib.parse import parse_qs, urlparse

from core.analytics.services import MetricsAggregator, PerformanceSnapshot, PerformanceTracker
from core.ops.contracts import HealthStatus


class IWebController(Protocol):
    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...


# Dashboard API adapter (itt csak Python objektumot ad vissza).
class DashboardAPI:
    def __init__(
        self,
        performance_tracker: PerformanceTracker,
        metrics_agg: MetricsAggregator,
        get_model_summary: Callable[[], dict] | None = None,
        get_positions_snapshot: Callable[[], dict] | None = None,
        get_simulation_results: Callable[[], dict] | None = None,
        get_wallet_history: Callable[[int], list[dict]] | None = None,
        get_health_status: Callable[[], HealthStatus] | None = None,
        get_risk_diagnostics: Callable[[], dict] | None = None,
        get_training_summary: Callable[[], dict] | None = None,
    ) -> None:
        self.performance_tracker = performance_tracker
        self.metrics_agg = metrics_agg
        self.get_model_summary_fn = get_model_summary
        self.get_positions_snapshot_fn = get_positions_snapshot
        self.get_simulation_results_fn = get_simulation_results
        self.get_wallet_history_fn = get_wallet_history
        self.get_health_status_fn = get_health_status
        self.get_risk_diagnostics = get_risk_diagnostics
        self.get_training_summary_fn = get_training_summary

    def get_performance(self) -> dict:
        return asdict(self.performance_tracker.snapshot())

    def get_metrics(self) -> dict:
        metrics = self.metrics_agg.export()
        if self.get_risk_diagnostics is not None:
            metrics["risk"] = self.get_risk_diagnostics()
        return metrics

    def get_risk(self) -> dict:
        if self.get_risk_diagnostics is None:
            return {"rule_hits": {}, "block_reason_hits": {}}
        return self.get_risk_diagnostics()

    def get_model_summary(self) -> dict:
        if self.get_model_summary_fn is None:
            return {}
        return self.get_model_summary_fn()

    def get_positions(self) -> dict:
        if self.get_positions_snapshot_fn is None:
            return {}
        return self.get_positions_snapshot_fn()

    def get_health(self) -> HealthStatus:
        if self.get_health_status_fn is None:
            return HealthStatus(ok=True, details={"env": "dev"})
        return self.get_health_status_fn()

    def get_simulation_results(self) -> dict:
        if self.get_simulation_results_fn is not None:
            return self.get_simulation_results_fn()
        return {
            "performance": asdict(self.performance_tracker.snapshot()),
            "positions": self.get_positions(),
            "risk": self.get_risk(),
        }

    def get_timeseries(self, limit: int = 300) -> dict:
        wallet_history = self.get_wallet_history_fn(limit) if self.get_wallet_history_fn is not None else []
        model = self.get_model_summary()
        recent_prob = model.get("recent_prob_up", [])
        recent_sigma = model.get("recent_sigma", [])
        recent_mu = model.get("recent_mu", [])
        recent_realized = model.get("recent_realized_return", [])
        recent_error = model.get("recent_pred_error", [])
        if isinstance(recent_prob, list):
            recent_prob = recent_prob[-limit:]
        if isinstance(recent_sigma, list):
            recent_sigma = recent_sigma[-limit:]
        if isinstance(recent_mu, list):
            recent_mu = recent_mu[-limit:]
        if isinstance(recent_realized, list):
            recent_realized = recent_realized[-limit:]
        if isinstance(recent_error, list):
            recent_error = recent_error[-limit:]
        return {
            "wallet_history": wallet_history,
            "recent_prob_up": recent_prob,
            "recent_sigma": recent_sigma,
            "recent_mu": recent_mu,
            "recent_realized_return": recent_realized,
            "recent_pred_error": recent_error,
            "ar_metrics": model.get("ar_metrics", {}),
        }

    def get_training_summary(self) -> dict:
        if self.get_training_summary_fn is None:
            return {
                "status": "unavailable",
                "last_result": None,
                "history": [],
            }
        return self.get_training_summary_fn()

    def get_analysis_bundle(self, limit: int = 500) -> dict:
        simulation = self.get_simulation_results()
        timeseries = self.get_timeseries(limit=limit)
        model = self.get_model_summary()
        performance = self.get_performance()
        risk = self.get_risk()
        training = self.get_training_summary()

        wallet_history = timeseries.get("wallet_history", [])
        if not isinstance(wallet_history, list):
            wallet_history = []

        def _numeric_list(values: object) -> list[float]:
            if not isinstance(values, list):
                return []
            output: list[float] = []
            for value in values:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    output.append(number)
            return output

        def _list_value(source: dict, key: str) -> list:
            value = source.get(key)
            return value if isinstance(value, list) else []

        timestamps: list[object] = []
        equity_curve: list[float] = []
        cash_curve: list[float] = []
        pnl_curve: list[float] = []
        for item in wallet_history:
            if not isinstance(item, dict):
                continue
            timestamps.append(item.get("ts"))
            equity_curve.append(float(item.get("equity") or 0.0))
            cash_curve.append(float(item.get("cash") or 0.0))
            pnl_curve.append(float(item.get("pnl") or 0.0))

        prob_up = _numeric_list(_list_value(model, "recent_prob_up") or _list_value(timeseries, "recent_prob_up"))
        sigma = _numeric_list(_list_value(model, "recent_sigma") or _list_value(timeseries, "recent_sigma"))
        mu = _numeric_list(_list_value(model, "recent_mu") or _list_value(timeseries, "recent_mu"))
        realized_return = _numeric_list(
            _list_value(model, "recent_realized_return") or _list_value(timeseries, "recent_realized_return")
        )
        pred_error = _numeric_list(_list_value(model, "recent_pred_error") or _list_value(timeseries, "recent_pred_error"))
        realized_up = [1 if value > 0 else 0 for value in realized_return]

        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "limit": limit,
            "simulation": simulation,
            "performance": performance,
            "risk": risk,
            "model": model,
            "training": training,
            "timeseries": timeseries,
            "curves": {
                "timestamps": timestamps,
                "equity": equity_curve,
                "cash": cash_curve,
                "pnl": pnl_curve,
                "prob_up": prob_up,
                "sigma": sigma,
                "mu": mu,
                "realized_return": realized_return,
                "realized_up": realized_up,
                "pred_error": pred_error,
            },
        }


# Custom ThreadingHTTPServer with SO_REUSEADDR to allow quick rebinding
class ReuseAddrThreadingHTTPServer(ThreadingHTTPServer):
    """HTTPServer that allows reusing addresses in TIME_WAIT state."""
    def server_bind(self) -> None:
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        super().server_bind()


# WebSocket streamer placeholder adapter.
class WebSocketStreamer:
    def __init__(self, performance_tracker: PerformanceTracker) -> None:
        self.clients: list[object] = []
        self.performance_tracker = performance_tracker

    async def broadcast(self, snapshot: PerformanceSnapshot) -> None:
        _ = snapshot

    def connect(self, client: object) -> None:
        self.clients.append(client)

    def disconnect(self, client: object) -> None:
        self.clients = [item for item in self.clients if item is not client]


# HTTP server placeholder.
class HttpServer:
    def __init__(self, api: DashboardAPI, ws: WebSocketStreamer, host: str = "127.0.0.1", port: int = 8000) -> None:
        self.api = api
        self.ws = ws
        self.started = False
        self.host = host
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _serialize(self, payload: object) -> bytes:
        safe_payload = self._sanitize_json(payload)
        return json.dumps(safe_payload, default=self._json_default, allow_nan=False).encode("utf-8")

    @classmethod
    def _sanitize_json(cls, value: object) -> object:
        if isinstance(value, dict):
            return {str(k): cls._sanitize_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._sanitize_json(v) for v in value]
        if isinstance(value, tuple):
            return [cls._sanitize_json(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    @staticmethod
    def _json_default(value: object) -> object:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, HealthStatus):
            return {"ok": value.ok, "details": value.details}
        return str(value)

    async def start(self) -> None:
        api = self.api
        serializer = self._serialize

        class _Handler(BaseHTTPRequestHandler):
            # Force HTTP/1.0 so the browser never attempts keep-alive reuse.
            # Python's BaseHTTPRequestHandler does not properly implement
            # HTTP/1.1 keep-alive, causing browsers to hang on the second
            # request until the TCP connection times out.
            protocol_version = "HTTP/1.0"

            def _send_no_cache_headers(self) -> None:
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.send_header("Connection", "close")

            def _send_body(self, code: int, content_type: str, body: bytes, *, cache: bool = False) -> None:
                """Send a complete HTTP response with Content-Length so the browser never waits for EOF."""
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                if cache:
                    self.send_header("Cache-Control", "public, max-age=86400")
                else:
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                try:
                    self._do_GET_inner()
                except Exception as exc:  # pragma: no cover
                    self.handle_error_in_handler(exc)

            def _do_GET_inner(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                query = parse_qs(parsed.query)

                if path == "/":
                    self._send_body(200, "text/html; charset=utf-8",
                                    _load_page_html("index.html").encode("utf-8"))
                    return

                if path in _MULTI_PAGE_ROUTES:
                    self._send_body(200, "text/html; charset=utf-8",
                                    _load_page_html(path[1:]).encode("utf-8"))
                    return

                # Serve local static assets (JS/CSS bundled locally)
                _STATIC_FILES = {
                    "/chart.umd.min.js": ("application/javascript", os.path.join(_DASHBOARD_DIR, "chart.umd.min.js")),
                    "/tailwind.js": ("application/javascript", os.path.join(_DASHBOARD_DIR, "tailwind.js")),
                    "/tailwind.min.css": ("text/css", os.path.join(_DASHBOARD_DIR, "tailwind.min.css")),
                    "/help.js": ("application/javascript", os.path.join(_DASHBOARD_DIR, "help.js")),
                    "/tooltip.js": ("application/javascript", os.path.join(_DASHBOARD_DIR, "tooltip.js")),
                }
                if path in _STATIC_FILES:
                    mime, fpath = _STATIC_FILES[path]
                    if os.path.exists(fpath):
                        with open(fpath, "rb") as sf:
                            data = sf.read()
                        self._send_body(200, mime, data, cache=True)
                    else:
                        self._send_body(404, "application/json", b'{"error":"not_found"}')
                    return

                routes = {
                    "/health": api.get_health,
                    "/performance": api.get_performance,
                    "/metrics": api.get_metrics,
                    "/risk": api.get_risk,
                    "/training": api.get_training_summary,
                    "/model": api.get_model_summary,
                    "/positions": api.get_positions,
                    "/simulation": api.get_simulation_results,
                }
                if path == "/timeseries":
                    raw_limit = query.get("limit", ["300"])[0]
                    try:
                        limit = max(10, min(2000, int(raw_limit)))
                    except ValueError:
                        limit = 300
                    self._send_body(200, "application/json", serializer(api.get_timeseries(limit=limit)))
                    return

                if path == "/analysis_bundle":
                    raw_limit = query.get("limit", ["500"])[0]
                    try:
                        limit = max(10, min(5000, int(raw_limit)))
                    except ValueError:
                        limit = 500
                    self._send_body(200, "application/json", serializer(api.get_analysis_bundle(limit=limit)))
                    return

                if path == "/benchmark":
                    payload = _benchmark_mod.get_state() if _benchmark_mod is not None else {"status": "unavailable"}
                    self._send_body(200, "application/json", serializer(payload))
                    return

                if path == "/ltsf_benchmark":
                    payload = _ltsf_mod.get_state() if _ltsf_mod is not None else {"status": "unavailable"}
                    self._send_body(200, "application/json", serializer(payload))
                    return

                handler = routes.get(path)
                if handler is None:
                    self._send_body(404, "application/json", serializer({"error": "not_found", "path": path}))
                    return

                self._send_body(200, "application/json", serializer(handler()))

            def do_POST(self) -> None:  # noqa: N802
                try:
                    self._do_POST_inner()
                except Exception as exc:  # pragma: no cover
                    self.handle_error_in_handler(exc)

            def _do_POST_inner(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path

                if path == "/benchmark/run":
                    if _benchmark_mod is None:
                        self._send_body(503, "application/json", serializer({"ok": False, "error": "benchmark module unavailable"}))
                    else:
                        started = _benchmark_mod.trigger()
                        self._send_body(200, "application/json", serializer({"ok": True, "started": started}))
                    return

                if path == "/ltsf_benchmark/run":
                    if _ltsf_mod is None:
                        self._send_body(503, "application/json", serializer({"ok": False, "error": "ltsf_benchmark module unavailable"}))
                    else:
                        content_len = int(self.headers.get("Content-Length", 0))
                        cfg: dict = {}
                        if content_len > 0:
                            raw = self.rfile.read(content_len)
                            try:
                                cfg = json.loads(raw)
                            except Exception:
                                cfg = {}
                        started = _ltsf_mod.trigger(cfg)
                        self._send_body(200, "application/json", serializer({"ok": True, "started": started}))
                    return

                self._send_body(404, "application/json", serializer({"error": "not_found", "path": path}))

            def log_message(self, format: str, *args: object) -> None:
                # Csendes HTTP handler, hogy a terminal ne legyen tele loggal.
                _ = format
                _ = args

            def log_error(self, format: str, *args: object) -> None:  # noqa: N802
                # Hibákat MINDIG logoljuk, még ha az access logot elnémítjuk is.
                import sys
                try:
                    msg = format % args
                except Exception:
                    msg = repr((format, args))
                path = getattr(self, "path", "<unknown>")
                print(f"[HTTP ERROR] {path!r}: {msg}", file=sys.stderr, flush=True)

            def handle_error_in_handler(self, exc: Exception) -> None:
                import sys, traceback
                path = getattr(self, "path", "<unknown>")
                print(f"[HTTP CRASH] {path!r}: {exc!r}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                try:
                    self._send_body(500, "text/plain", f"Internal error: {exc}".encode())
                except Exception:
                    pass

        try:
            self._httpd = ReuseAddrThreadingHTTPServer((self.host, self.port), _Handler)
        except OSError as error:
            raise RuntimeError(
                f"HTTP server bind failed on {self.host}:{self.port}. "
                f"Address already in use or restricted. Set a different port via config/env."
            ) from error
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        self.started = True

    async def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.started = False


def _load_dashboard_html() -> str:
    """Load dashboard HTML from the separate file, fall back to embedded string."""
    html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return _DASHBOARD_HTML


_DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "dashboard")
_MULTI_PAGE_ROUTES = frozenset({"/model.html", "/signals.html", "/training.html", "/benchmark.html", "/ltsf_benchmark.html", "/kla_kalman.html", "/complexity.html"})

try:
    from adapters.offline_training import benchmark as _benchmark_mod
except Exception:  # pragma: no cover
    _benchmark_mod = None  # type: ignore[assignment]

try:
    from adapters.offline_training import ltsf_benchmark as _ltsf_mod
except Exception:  # pragma: no cover
    _ltsf_mod = None  # type: ignore[assignment]


def _load_page_html(page: str) -> str:
    """Load a named HTML page from the dashboard/ subfolder."""
    page_path = os.path.join(_DASHBOARD_DIR, page)
    if os.path.exists(page_path):
        with open(page_path, "r", encoding="utf-8") as fh:
            return fh.read()
    if page == "index.html":
        return _load_dashboard_html()
    return (
        "<!doctype html><html><body style='background:#080c14;color:#e2e8f0;"
        f"font-family:monospace;padding:2rem'><h1>404: {page} nem található</h1></body></html>"
    )


_DASHBOARD_HTML_LEGACY = """<!doctype html>
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>🚀 Trading Simulation Dashboard</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: #e2e8f0;
                padding: 20px;
                min-height: 100vh;
            }
            .container { max-width: 1400px; margin: 0 auto; }
            h1 { 
                font-size: 32px;
                font-weight: 700;
                margin-bottom: 8px;
                background: linear-gradient(90deg, #60a5fa, #a78bfa);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .subtitle { color: #94a3b8; margin-bottom: 24px; font-size: 14px; }
            .controls { 
                background: rgba(30, 41, 59, 0.5);
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 24px;
                backdrop-filter: blur(10px);
            }
            .controls label { 
                color: #cbd5e1;
                font-size: 14px;
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                gap: 8px;
            }
            .controls input[type="checkbox"] {
                width: 18px;
                height: 18px;
                cursor: pointer;
            }
            .status { 
                display: inline-block;
                margin-left: 16px;
                padding: 4px 12px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
            }
            .status.on { background: rgba(34, 197, 94, 0.2); color: #4ade80; }
            .status.off { background: rgba(239, 68, 68, 0.2); color: #f87171; }
            .metrics { 
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
                margin-bottom: 24px;
            }
            .metric-card { 
                background: rgba(30, 41, 59, 0.6);
                border: 1px solid rgba(148, 163, 184, 0.15);
                border-radius: 16px;
                padding: 20px;
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            }
            .metric-card:hover {
                transform: translateY(-2px);
                border-color: rgba(148, 163, 184, 0.3);
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
            }
            .metric-label { 
                font-size: 12px;
                color: #94a3b8;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 8px;
            }
            .metric-value { 
                font-size: 28px;
                font-weight: 700;
                color: #e2e8f0;
            }
            .metric-value.positive { color: #4ade80; }
            .metric-value.negative { color: #f87171; }
            .charts-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                gap: 20px;
                margin-bottom: 24px;
            }
            .chart-container {
                background: rgba(30, 41, 59, 0.6);
                border: 1px solid rgba(148, 163, 184, 0.15);
                border-radius: 16px;
                padding: 20px;
                backdrop-filter: blur(10px);
            }
            .chart-title {
                font-size: 16px;
                font-weight: 600;
                color: #cbd5e1;
                margin-bottom: 16px;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .chart-title::before {
                content: '';
                width: 4px;
                height: 20px;
                background: linear-gradient(180deg, #60a5fa, #a78bfa);
                border-radius: 2px;
            }
            canvas { 
                width: 100% !important;
                height: 280px !important;
                border-radius: 12px;
            }
            .json-container {
                background: rgba(17, 24, 39, 0.8);
                border: 1px solid rgba(148, 163, 184, 0.15);
                border-radius: 16px;
                padding: 20px;
                margin-top: 24px;
                backdrop-filter: blur(10px);
            }
            pre { 
                color: #e2e8f0;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                overflow: auto;
                max-height: 400px;
                line-height: 1.6;
            }
            .risk-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                gap: 16px;
                margin-bottom: 24px;
            }
            .risk-box {
                background: rgba(30, 41, 59, 0.6);
                border: 1px solid rgba(148, 163, 184, 0.15);
                border-radius: 16px;
                padding: 16px;
                backdrop-filter: blur(10px);
            }
            .risk-box h3 {
                font-size: 14px;
                color: #cbd5e1;
                margin: 0 0 10px 0;
            }
            .badge {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
                margin-left: 8px;
            }
            .badge.live { background: rgba(34, 197, 94, 0.2); color: #4ade80; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Trading Simulation Dashboard</h1>
            <div class="subtitle">Real-time monitoring · Automatikus frissítés 2mp-enként <span class="badge live">LIVE</span></div>

            <div class="controls">
                <label>
                    <input type="checkbox" id="kalmanToggle" />
                    Kalman Filter bekapcsolva
                </label>
                <span id="kalmanStatus" class="status"></span>
            </div>

            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-label">💰 Kezdőtőke</div>
                    <div class="metric-value" id="initialCash">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">💵 Cash</div>
                    <div class="metric-value" id="cash">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">📊 Equity</div>
                    <div class="metric-value" id="equity">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">📈 PnL</div>
                    <div class="metric-value" id="pnl">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">⚡ Sharpe</div>
                    <div class="metric-value" id="sharpe">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">📉 Drawdown</div>
                    <div class="metric-value" id="drawdown">-</div>
                </div>
            </div>

            <div class="charts-grid">
                <div class="chart-container">
                    <div class="chart-title">💰 Wallet Equity Curve</div>
                    <canvas id="equityChart"></canvas>
                </div>
                <div class="chart-container">
                    <div class="chart-title">💵 Cash Balance</div>
                    <canvas id="cashChart"></canvas>
                </div>
            </div>

            <div class="charts-grid">
                <div class="chart-container">
                    <div class="chart-title">🤖 Model Prediction (prob_up)</div>
                    <canvas id="probChart"></canvas>
                </div>
                <div class="chart-container">
                    <div class="chart-title">📊 PnL Distribution</div>
                    <canvas id="pnlChart"></canvas>
                </div>
            </div>

            <div class="chart-container" style="margin-bottom: 24px;">
                <div class="chart-title">🛡️ Risk Diagnostics</div>
                <div class="risk-grid">
                    <div class="risk-box">
                        <h3>Rule Hits</h3>
                        <pre id="riskRuleHits">Betöltés...</pre>
                    </div>
                    <div class="risk-box">
                        <h3>Block Reason Hits</h3>
                        <pre id="riskBlockReasons">Betöltés...</pre>
                    </div>
                </div>
            </div>

            <div class="json-container">
                <div class="chart-title">📋 Raw Data (JSON)</div>
                <pre id="jsonOut">Betöltés...</pre>
            </div>
        </div>

        <script>
            function fmt(value) {
                if (typeof value !== 'number') return '-';
                return value.toLocaleString('hu-HU', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
            }

            function sortCounterObject(obj) {
                const entries = Object.entries(obj || {});
                entries.sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0));
                return Object.fromEntries(entries);
            }

            function drawLineChart(canvasId, values, color, label, minY = null, maxY = null, showGrid = true) {
                const canvas = document.getElementById(canvasId);
                if (!canvas) return;
                const ctx = canvas.getContext('2d');
                const w = canvas.width;
                const h = canvas.height;
                ctx.clearRect(0, 0, w, h);

                if (!values || values.length < 2) {
                    ctx.fillStyle = '#64748b';
                    ctx.font = '14px sans-serif';
                    ctx.fillText('Nincs elég adat a megjelenítéshez', w/2 - 100, h/2);
                    return;
                }

                const pad = 40;
                const min = minY ?? Math.min(...values);
                const max = maxY ?? Math.max(...values);
                const range = Math.max(1e-9, max - min);

                // Grid
                if (showGrid) {
                    ctx.strokeStyle = 'rgba(148, 163, 184, 0.1)';
                    ctx.lineWidth = 1;
                    for (let i = 0; i <= 4; i++) {
                        const y = pad + (i * (h - 2 * pad)) / 4;
                        ctx.beginPath();
                        ctx.moveTo(pad, y);
                        ctx.lineTo(w - pad, y);
                        ctx.stroke();
                    }
                }

                // Axes
                ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(pad, h - pad);
                ctx.lineTo(w - pad, h - pad);
                ctx.moveTo(pad, pad);
                ctx.lineTo(pad, h - pad);
                ctx.stroke();

                // Labels
                ctx.fillStyle = '#94a3b8';
                ctx.font = '11px sans-serif';
                ctx.fillText(max.toFixed(2), 5, pad + 10);
                ctx.fillText(min.toFixed(2), 5, h - pad - 5);

                // Line
                const gradient = ctx.createLinearGradient(0, 0, 0, h);
                gradient.addColorStop(0, color);
                gradient.addColorStop(1, color + '80');
                ctx.strokeStyle = gradient;
                ctx.lineWidth = 3;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.beginPath();
                
                values.forEach((v, i) => {
                    const x = pad + (i * (w - 2 * pad)) / (values.length - 1);
                    const y = h - pad - ((v - min) * (h - 2 * pad)) / range;
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                });
                ctx.stroke();

                // Area fill
                ctx.lineTo(w - pad, h - pad);
                ctx.lineTo(pad, h - pad);
                ctx.closePath();
                const areaGradient = ctx.createLinearGradient(0, 0, 0, h);
                areaGradient.addColorStop(0, color + '40');
                areaGradient.addColorStop(1, color + '00');
                ctx.fillStyle = areaGradient;
                ctx.fill();

                // Data points on hover (just show count)
                ctx.fillStyle = '#cbd5e1';
                ctx.font = '12px sans-serif';
                ctx.fillText(`${values.length} adatpont`, w - pad - 80, pad - 10);
            }

            async function syncKalmanConfig() {
                try {
                    const response = await fetch('/config');
                    const cfg = await response.json();
                    const toggle = document.getElementById('kalmanToggle');
                    toggle.checked = !!cfg.use_kalman;
                    const status = document.getElementById('kalmanStatus');
                    status.textContent = cfg.use_kalman ? '✓ Aktív' : '✗ Kikapcsolva';
                    status.className = cfg.use_kalman ? 'status on' : 'status off';
                } catch (_) {}
            }

            async function updateKalman(enabled) {
                try {
                    const response = await fetch('/config/use_kalman', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ enabled })
                    });
                    const cfg = await response.json();
                    const status = document.getElementById('kalmanStatus');
                    status.textContent = cfg.use_kalman ? '✓ Aktív' : '✗ Kikapcsolva';
                    status.className = cfg.use_kalman ? 'status on' : 'status off';
                } catch (err) {
                    console.error('Kalman update error:', err);
                }
            }

            async function refresh() {
                try {
                    const [simRes, tsRes, modelRes, perfRes, riskRes] = await Promise.all([
                        fetch('/simulation'),
                        fetch('/timeseries?limit=500'),
                        fetch('/model'),
                        fetch('/performance'),
                        fetch('/risk')
                    ]);
                    const data = await simRes.json();
                    const tsData = await tsRes.json();
                    const modelData = await modelRes.json();
                    const perfData = await perfRes.json();
                    const riskData = await riskRes.json();

                    // Update metrics
                    const wallet = data.wallet || {};
                    const performance = data.performance || perfData || {};
                    
                    document.getElementById('initialCash').textContent = fmt(wallet.initial_cash || 10000);
                    
                    const cashEl = document.getElementById('cash');
                    cashEl.textContent = fmt(wallet.cash || 0);
                    cashEl.className = 'metric-value ' + ((wallet.cash || 0) >= 0 ? 'positive' : 'negative');
                    
                    const equityEl = document.getElementById('equity');
                    equityEl.textContent = fmt(wallet.equity || 0);
                    equityEl.className = 'metric-value ' + ((wallet.equity || 0) >= 10000 ? 'positive' : 'negative');
                    
                    const pnlEl = document.getElementById('pnl');
                    pnlEl.textContent = fmt(wallet.pnl || 0);
                    pnlEl.className = 'metric-value ' + ((wallet.pnl || 0) >= 0 ? 'positive' : 'negative');

                    document.getElementById('sharpe').textContent = fmt(performance.sharpe || 0);
                    
                    const ddEl = document.getElementById('drawdown');
                    ddEl.textContent = fmt(performance.drawdown || 0) + '%';
                    ddEl.className = 'metric-value ' + ((performance.drawdown || 0) <= 10 ? 'positive' : 'negative');

                    // Charts
                    const walletHistory = tsData.wallet_history || [];
                    const equityValues = walletHistory.map(x => Number(x.equity || 0));
                    const cashValues = walletHistory.map(x => Number(x.cash || 0));
                    const pnlValues = walletHistory.map(x => Number(x.pnl || 0));

                    drawLineChart('equityChart', equityValues, '#3b82f6', 'Equity', null, null, true);
                    drawLineChart('cashChart', cashValues, '#10b981', 'Cash', null, null, true);
                    drawLineChart('pnlChart', pnlValues, '#f59e0b', 'PnL', null, null, true);

                    const probs = modelData.recent_prob_up || tsData.recent_prob_up || [];
                    drawLineChart('probChart', probs.map(x => Number(x)), '#a78bfa', 'prob_up', 0, 1, true);

                    const risk = data.risk || riskData || {};
                    const ruleHits = sortCounterObject(risk.rule_hits || {});
                    const blockReasonHits = sortCounterObject(risk.block_reason_hits || {});
                    document.getElementById('riskRuleHits').textContent = JSON.stringify(ruleHits, null, 2);
                    document.getElementById('riskBlockReasons').textContent = JSON.stringify(blockReasonHits, null, 2);

                    document.getElementById('jsonOut').textContent = JSON.stringify(data, null, 2);
                } catch (err) {
                    document.getElementById('jsonOut').textContent = 'Hiba a lekérésénél: ' + err;
                    const ruleEl = document.getElementById('riskRuleHits');
                    const reasonEl = document.getElementById('riskBlockReasons');
                    if (ruleEl) ruleEl.textContent = 'Hiba a risk adatok lekérésénél';
                    if (reasonEl) reasonEl.textContent = 'Hiba a risk adatok lekérésénél';
                    console.error('Refresh error:', err);
                }
            }

            document.getElementById('kalmanToggle').addEventListener('change', (event) => {
                updateKalman(event.target.checked);
            });

            syncKalmanConfig();
            refresh();
            setInterval(refresh, 2000);
        </script>
    </body>
</html>
"""


_DASHBOARD_HTML = """<!doctype html>
<html lang="hu">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Trading Control Dashboard</title>
        <style>
            @import url("https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap");

            :root {
                --bg: #f4f8ff;
                --panel: #ffffff;
                --line: #d5e2ff;
                --text: #17223a;
                --muted: #617091;
                --brand: #1463ff;
                --brand-2: #22b8cf;
                --good: #12805a;
                --bad: #b93f3f;
                --shadow: 0 10px 28px rgba(20, 53, 129, 0.09);
            }

            * { box-sizing: border-box; }
            html, body { margin: 0; }

            body {
                font-family: "Manrope", sans-serif;
                color: var(--text);
                background:
                    radial-gradient(1200px 500px at 10% -10%, rgba(20, 99, 255, 0.16), transparent 70%),
                    radial-gradient(900px 500px at 90% -20%, rgba(34, 184, 207, 0.20), transparent 70%),
                    var(--bg);
            }

            .container {
                max-width: 1380px;
                margin: 0 auto;
                padding: 22px;
            }

            .hero {
                background: linear-gradient(120deg, #ffffff 0%, #eef4ff 68%, #e8fbff 100%);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 18px 22px;
                box-shadow: var(--shadow);
                display: grid;
                grid-template-columns: 1fr auto;
                gap: 14px;
                align-items: center;
            }

            .hero h1 {
                margin: 0;
                font-size: 30px;
                letter-spacing: 0.3px;
            }

            .hero p {
                margin: 5px 0 0;
                color: var(--muted);
            }

            .kalman-row {
                display: flex;
                align-items: center;
                gap: 12px;
                font-size: 14px;
            }

            .status {
                border-radius: 999px;
                padding: 5px 11px;
                font-size: 12px;
                font-weight: 800;
                letter-spacing: 0.4px;
            }

            .status.on { color: #0f6c49; background: #d9f6ec; }
            .status.off { color: #8d3232; background: #ffe3e3; }

            .top-nav {
                margin-top: 16px;
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }

            .tab-btn {
                border: 1px solid var(--line);
                border-radius: 12px;
                padding: 10px 14px;
                background: #fff;
                color: var(--text);
                font-weight: 700;
                cursor: pointer;
                transition: transform 0.18s ease, background 0.18s ease, border-color 0.18s ease;
            }

            .tab-btn:hover { transform: translateY(-1px); background: #f6f9ff; }
            .tab-btn.active { background: #e9f1ff; border-color: #b8ceff; color: #0d4ad0; }

            .view {
                margin-top: 16px;
                display: none;
                animation: fadeIn 180ms ease;
            }

            .view.active { display: block; }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(4px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
                gap: 10px;
            }

            .card {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 12px;
                box-shadow: var(--shadow);
            }

            .label {
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.6px;
                color: var(--muted);
            }

            .value {
                margin-top: 6px;
                font-size: 27px;
                font-weight: 800;
            }

            .pos { color: var(--good); }
            .neg { color: var(--bad); }

            .grid-2 {
                margin-top: 12px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(430px, 1fr));
                gap: 12px;
            }

            .panel {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 12px;
                box-shadow: var(--shadow);
            }

            .title {
                margin: 0 0 10px;
                font-size: 14px;
                font-weight: 800;
                color: #28406f;
            }

            canvas { width: 100% !important; height: 280px !important; border-radius: 10px; }

            .model-list {
                margin-top: 10px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 10px;
            }

            .model-chip {
                border: 1px solid #bfd4ff;
                border-radius: 12px;
                padding: 10px;
                background: linear-gradient(145deg, #f5f9ff 0%, #eff5ff 100%);
            }

            .model-chip h4 {
                margin: 0;
                font-size: 13px;
            }

            .mono {
                font-family: "IBM Plex Mono", monospace;
                font-size: 12px;
                color: #36486e;
            }

            pre {
                margin: 0;
                padding: 10px;
                background: #f7faff;
                border: 1px solid #d8e6ff;
                border-radius: 12px;
                font-family: "IBM Plex Mono", monospace;
                color: #23365f;
                max-height: 360px;
                overflow: auto;
            }

            .analysis-head {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 8px;
                flex-wrap: wrap;
            }

            .analysis-actions {
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
            }

            .ghost-btn {
                border: 1px solid #bfd4ff;
                border-radius: 10px;
                padding: 8px 12px;
                background: #f6faff;
                color: #1d3b77;
                font-weight: 700;
                cursor: pointer;
            }

            .ghost-btn:hover {
                background: #eaf3ff;
            }

            .hint {
                margin: 8px 0 10px;
                color: var(--muted);
                font-size: 12px;
                font-weight: 600;
            }

            @media (max-width: 880px) {
                .hero { grid-template-columns: 1fr; }
                .grid-2 { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <section class="hero">
                <div>
                    <h1>Trading Control Dashboard</h1>
                    <p>Vilagos, felsomenus nezetvaltas: attekintes, modellek, kockazat, nyers adatok.</p>
                </div>
            </section>

            <nav class="top-nav">
                <button class="tab-btn active" data-view="overview">Attekintes</button>
                <button class="tab-btn" data-view="models">Modellek</button>
                <button class="tab-btn" data-view="training">Training</button>
                <button class="tab-btn" data-view="risk">Kockazat</button>
                <button class="tab-btn" data-view="raw">Raw</button>
                <button class="tab-btn" data-view="analysis">Elemzes Export</button>
            </nav>

            <section id="view-overview" class="view active">
                <div class="kpi-grid">
                    <div class="card"><div class="label">Equity</div><div class="value" id="equity">-</div></div>
                    <div class="card"><div class="label">PnL</div><div class="value" id="pnl">-</div></div>
                    <div class="card"><div class="label">Sharpe</div><div class="value" id="sharpe">-</div></div>
                    <div class="card"><div class="label">Drawdown</div><div class="value" id="drawdown">-</div></div>
                    <div class="card"><div class="label">AR Accuracy</div><div class="value" id="arAcc">-</div></div>
                    <div class="card"><div class="label">AR Samples</div><div class="value" id="arSamples">-</div></div>
                </div>
                <div class="grid-2">
                    <div class="panel">
                        <h3 class="title">Equity Curve</h3>
                        <canvas id="equityChart"></canvas>
                    </div>
                    <div class="panel">
                        <h3 class="title">Probability vs Realized Up</h3>
                        <canvas id="probChart"></canvas>
                    </div>
                </div>
            </section>

            <section id="view-models" class="view">
                <div class="kpi-grid">
                    <div class="card"><div class="label">AR MAE</div><div class="value" id="arMae">-</div></div>
                    <div class="card"><div class="label">AR RMSE</div><div class="value" id="arRmse">-</div></div>
                    <div class="card"><div class="label">Brier</div><div class="value" id="arBrier">-</div></div>
                    <div class="card"><div class="label">Model Version</div><div class="value mono" id="modelVersion">unknown</div></div>
                </div>
                <div class="grid-2">
                    <div class="panel">
                        <h3 class="title">Predicted Return (mu) vs Realized Return</h3>
                        <canvas id="muChart"></canvas>
                    </div>
                    <div class="panel">
                        <h3 class="title">Prediction Error (mu - realized)</h3>
                        <canvas id="errChart"></canvas>
                    </div>
                </div>
                <div class="panel" style="margin-top: 12px;">
                    <h3 class="title">Felismerett Modellek / komponensek</h3>
                    <div id="modelList" class="model-list"></div>
                </div>
            </section>

            <section id="view-risk" class="view">
                <div class="grid-2">
                    <div class="panel">
                        <h3 class="title">Risk Diagnostics</h3>
                        <pre id="riskOut">Loading...</pre>
                    </div>
                    <div class="panel">
                        <h3 class="title">Model Summary</h3>
                        <pre id="modelOut">Loading...</pre>
                    </div>
                </div>
            </section>

            <section id="view-training" class="view">
                <div class="grid-2">
                    <div class="panel">
                        <h3 class="title">Training Status / Last Result</h3>
                        <pre id="trainingOut">Loading...</pre>
                    </div>
                    <div class="panel">
                        <h3 class="title">Training History</h3>
                        <pre id="trainingHistoryOut">Loading...</pre>
                    </div>
                </div>
            </section>

            <section id="view-raw" class="view">
                <div class="panel">
                    <h3 class="title">Simulation JSON</h3>
                    <pre id="rawOut">Loading...</pre>
                </div>
            </section>

            <section id="view-analysis" class="view">
                <div class="panel">
                    <div class="analysis-head">
                        <h3 class="title">Osszesitett adatexport minden nezetbol</h3>
                        <div class="analysis-actions">
                            <button id="analysisRefreshBtn" class="ghost-btn">Frissites</button>
                            <button id="analysisCopyBtn" class="ghost-btn">Masolas vagolapra</button>
                        </div>
                    </div>
                    <p id="analysisStatus" class="hint">Varakozas az elso adathalmazra...</p>
                    <pre id="analysisOut">Loading...</pre>
                </div>
            </section>
        </div>

        <script>
            let latestAnalysisText = "";

            function fmt(v, digits = 2) {
                if (typeof v !== "number" || Number.isNaN(v)) return "-";
                return v.toLocaleString("hu-HU", { minimumFractionDigits: digits, maximumFractionDigits: digits });
            }

            function pct(v) {
                if (typeof v !== "number" || Number.isNaN(v)) return "-";
                return (v * 100).toLocaleString("hu-HU", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + "%";
            }

            function setValue(id, value, positiveIsGood = null) {
                const el = document.getElementById(id);
                if (!el) return;
                el.textContent = value;
                el.className = "value";
                if (positiveIsGood === true) el.classList.add("pos");
                if (positiveIsGood === false) el.classList.add("neg");
            }

            function drawLine(canvasId, values, color, minY = null, maxY = null) {
                const canvas = document.getElementById(canvasId);
                if (!canvas) return;
                const ctx = canvas.getContext("2d");
                const w = canvas.width;
                const h = canvas.height;
                ctx.clearRect(0, 0, w, h);

                if (!values || values.length < 2) {
                    ctx.fillStyle = "#6b7b9f";
                    ctx.font = "13px Manrope";
                    ctx.fillText("Nincs eleg adat", 18, 24);
                    return;
                }

                const pad = 32;
                const min = minY ?? Math.min(...values);
                const max = maxY ?? Math.max(...values);
                const range = Math.max(max - min, 1e-9);

                ctx.strokeStyle = "rgba(94, 123, 186, 0.22)";
                for (let i = 0; i <= 4; i++) {
                    const y = pad + ((h - 2 * pad) * i) / 4;
                    ctx.beginPath();
                    ctx.moveTo(pad, y);
                    ctx.lineTo(w - pad, y);
                    ctx.stroke();
                }

                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.beginPath();
                values.forEach((v, i) => {
                    const x = pad + ((w - 2 * pad) * i) / (values.length - 1);
                    const y = h - pad - ((v - min) * (h - 2 * pad)) / range;
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                });
                ctx.stroke();
            }

            function drawDual(canvasId, a, colorA, b, colorB, minY = null, maxY = null) {
                const valuesA = (a || []).map((x) => Number(x));
                const valuesB = (b || []).map((x) => Number(x));
                const n = Math.min(valuesA.length, valuesB.length);
                if (n < 2) {
                    drawLine(canvasId, valuesA, colorA, minY, maxY);
                    return;
                }

                const aSlice = valuesA.slice(-n);
                const bSlice = valuesB.slice(-n);
                drawLine(canvasId, aSlice, colorA, minY, maxY);

                const canvas = document.getElementById(canvasId);
                const ctx = canvas.getContext("2d");
                const w = canvas.width;
                const h = canvas.height;
                const pad = 32;
                const min = minY ?? Math.min(...aSlice, ...bSlice);
                const max = maxY ?? Math.max(...aSlice, ...bSlice);
                const range = Math.max(max - min, 1e-9);

                ctx.strokeStyle = colorB;
                ctx.lineWidth = 2;
                ctx.beginPath();
                bSlice.forEach((v, i) => {
                    const x = pad + ((w - 2 * pad) * i) / (n - 1);
                    const y = h - pad - ((v - min) * (h - 2 * pad)) / range;
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                });
                ctx.stroke();
            }

            function activateView(name) {
                document.querySelectorAll(".tab-btn").forEach((btn) => {
                    btn.classList.toggle("active", btn.dataset.view === name);
                });
                document.querySelectorAll(".view").forEach((section) => {
                    section.classList.toggle("active", section.id === `view-${name}`);
                });
            }

            function renderModelList(model) {
                const el = document.getElementById("modelList");
                if (!el) return;

                const items = [
                    { name: "Predictor", value: model.new_version || model.current_version || "unknown" },
                    { name: "Regime", value: Array.isArray(model.recent_mu) && model.recent_mu.length > 0 ? "streaming" : "idle" },
                    { name: "Feature Drift", value: model.feature_drift_score ?? "n/a" },
                    { name: "Prob Sample Count", value: (model.recent_prob_up || []).length },
                ];

                el.innerHTML = items.map((item) =>
                    `<div class="model-chip"><h4>${item.name}</h4><div class="mono">${String(item.value)}</div></div>`
                ).join("");
            }

            function normalizeNumberList(values) {
                if (!Array.isArray(values)) return [];
                return values.map((value) => Number(value)).filter((value) => Number.isFinite(value));
            }

            function buildAnalysisBundle(sim, ts, model, perf, risk, training) {
                const walletHistory = Array.isArray(ts.wallet_history) ? ts.wallet_history : [];
                const curves = {
                    timestamps: walletHistory.map((point) => point?.timestamp || null),
                    equity: walletHistory.map((point) => Number(point?.equity || 0)),
                    cash: walletHistory.map((point) => Number(point?.cash || 0)),
                    pnl: walletHistory.map((point) => Number(point?.pnl || 0)),
                    prob_up: normalizeNumberList(model.recent_prob_up || ts.recent_prob_up || []),
                    sigma: normalizeNumberList(model.recent_sigma || ts.recent_sigma || []),
                    mu: normalizeNumberList(model.recent_mu || ts.recent_mu || []),
                    realized_return: normalizeNumberList(model.recent_realized_return || ts.recent_realized_return || []),
                    pred_error: normalizeNumberList(model.recent_pred_error || ts.recent_pred_error || []),
                };
                curves.realized_up = curves.realized_return.map((value) => (value > 0 ? 1 : 0));

                return {
                    generated_at: new Date().toISOString(),
                    simulation: sim,
                    performance: perf,
                    risk,
                    model,
                    training,
                    timeseries: ts,
                    curves,
                };
            }

            function setAnalysisStatus(message, isError = false) {
                const el = document.getElementById("analysisStatus");
                if (!el) return;
                el.textContent = message;
                el.style.color = isError ? "#b93f3f" : "#617091";
            }

            function renderAnalysisBundle(bundle) {
                const out = document.getElementById("analysisOut");
                if (!out) return;
                latestAnalysisText = JSON.stringify(bundle, null, 2);
                out.textContent = latestAnalysisText;
                const lines = latestAnalysisText.split("\\n").length;
                setAnalysisStatus(`Frissitve: ${new Date().toLocaleString("hu-HU")} | Sorok: ${lines}`);
            }

            async function copyAnalysisBundle() {
                if (!latestAnalysisText) {
                    setAnalysisStatus("Nincs mit masolni. Varj egy frissitesre.", true);
                    return;
                }
                try {
                    if (navigator.clipboard && navigator.clipboard.writeText) {
                        await navigator.clipboard.writeText(latestAnalysisText);
                    } else {
                        const textarea = document.createElement("textarea");
                        textarea.value = latestAnalysisText;
                        textarea.setAttribute("readonly", "");
                        textarea.style.position = "fixed";
                        textarea.style.opacity = "0";
                        document.body.appendChild(textarea);
                        textarea.select();
                        const copied = document.execCommand("copy");
                        document.body.removeChild(textarea);
                        if (!copied) {
                            throw new Error("Copy failed");
                        }
                    }
                    setAnalysisStatus("Masolva a vagolapra.");
                } catch (_) {
                    setAnalysisStatus("A masolas nem sikerult. Jelold ki a blokkot es masold kezzel.", true);
                }
            }

            async function safeFetchJson(url, fallback = {}) {
                try {
                    const res = await fetch(url);
                    if (!res.ok) return fallback;
                    return await res.json();
                } catch (_) {
                    return fallback;
                }
            }

            async function refresh() {
                try {
                    const [sim, ts, model, perf, risk, training, bundle] = await Promise.all([
                        safeFetchJson("/simulation", {}),
                        safeFetchJson("/timeseries?limit=500", {}),
                        safeFetchJson("/model", {}),
                        safeFetchJson("/performance", {}),
                        safeFetchJson("/risk", {}),
                        safeFetchJson("/training", {}),
                        safeFetchJson("/analysis_bundle?limit=500", {})
                    ]);

                    const wallet = sim.wallet || {};
                    const performance = sim.performance || perf || {};
                    const ar = model.ar_metrics || ts.ar_metrics || {};

                    setValue("equity", fmt(Number(wallet.equity || 0)), Number(wallet.equity || 0) >= Number(wallet.initial_cash || 0));
                    setValue("pnl", fmt(Number(wallet.pnl || 0)), Number(wallet.pnl || 0) >= 0);
                    setValue("sharpe", fmt(Number(performance.sharpe || 0), 3), Number(performance.sharpe || 0) >= 0);
                    setValue("drawdown", fmt(Number(performance.drawdown || 0), 2) + "%", Number(performance.drawdown || 0) <= 15);

                    setValue("arAcc", pct(Number(ar.directional_accuracy || 0)), Number(ar.directional_accuracy || 0) >= 0.5);
                    setValue("arSamples", fmt(Number(ar.samples || 0), 0), Number(ar.samples || 0) > 50);
                    setValue("arMae", fmt(Number(ar.mae_return || 0), 5), Number(ar.mae_return || 0) <= 0.002);
                    setValue("arRmse", fmt(Number(ar.rmse_return || 0), 5), Number(ar.rmse_return || 0) <= 0.003);
                    setValue("arBrier", fmt(Number(ar.brier_up || 0), 4), Number(ar.brier_up || 0) <= 0.25);

                    const modelVersionEl = document.getElementById("modelVersion");
                    if (modelVersionEl) modelVersionEl.textContent = String(model.new_version || model.current_version || "unknown");

                    const walletHistory = ts.wallet_history || [];
                    const equity = walletHistory.map((x) => Number(x.equity || 0));
                    drawLine("equityChart", equity, "#1463ff");

                    const prob = (model.recent_prob_up || ts.recent_prob_up || []).map((x) => Number(x));
                    const realizedRet = (model.recent_realized_return || ts.recent_realized_return || []).map((x) => Number(x));
                    const realizedUp = realizedRet.map((x) => (x > 0 ? 1 : 0));
                    drawDual("probChart", prob, "#1c7ed6", realizedUp, "#099268", 0, 1);

                    const mu = (model.recent_mu || ts.recent_mu || []).map((x) => Number(x));
                    drawDual("muChart", mu, "#6741d9", realizedRet, "#e8590c");

                    const err = (model.recent_pred_error || ts.recent_pred_error || []).map((x) => Number(x));
                    drawLine("errChart", err, "#c2255c");

                    renderModelList(model || {});

                    const riskOut = document.getElementById("riskOut");
                    if (riskOut) riskOut.textContent = JSON.stringify(risk, null, 2);
                    const modelOut = document.getElementById("modelOut");
                    if (modelOut) modelOut.textContent = JSON.stringify(model, null, 2);
                    const trainingOut = document.getElementById("trainingOut");
                    if (trainingOut) {
                        const payload = {
                            status: training.status || "unknown",
                            started_at: training.started_at || null,
                            finished_at: training.finished_at || null,
                            last_result: training.last_result || null,
                        };
                        trainingOut.textContent = JSON.stringify(payload, null, 2);
                    }
                    const trainingHistoryOut = document.getElementById("trainingHistoryOut");
                    if (trainingHistoryOut) trainingHistoryOut.textContent = JSON.stringify(training.history || [], null, 2);
                    const rawOut = document.getElementById("rawOut");
                    if (rawOut) rawOut.textContent = JSON.stringify(sim, null, 2);

                    const mergedBundle = (bundle && Object.keys(bundle).length > 0)
                        ? bundle
                        : buildAnalysisBundle(sim, ts, model, performance, risk, training);
                    renderAnalysisBundle(mergedBundle);
                } catch (err) {
                    const modelOut = document.getElementById("modelOut");
                    if (modelOut) modelOut.textContent = "Refresh error: " + err;
                    setAnalysisStatus("Export frissitesi hiba: " + err, true);
                }
            }

            document.querySelectorAll(".tab-btn").forEach((btn) => {
                btn.addEventListener("click", () => activateView(btn.dataset.view));
            });

            const analysisRefreshBtn = document.getElementById("analysisRefreshBtn");
            if (analysisRefreshBtn) analysisRefreshBtn.addEventListener("click", refresh);
            const analysisCopyBtn = document.getElementById("analysisCopyBtn");
            if (analysisCopyBtn) analysisCopyBtn.addEventListener("click", copyAnalysisBundle);

            refresh();
            setInterval(refresh, 2000);
        </script>
    </body>
</html>
"""
