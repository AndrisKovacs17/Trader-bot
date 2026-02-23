from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from typing import Callable, Protocol
from urllib.parse import parse_qs, urlparse

from core.analytics.services import MetricsAggregator, ModelDiagnosticsService, PerformanceSnapshot, PerformanceTracker
from core.application.stores import Config, PositionStore, SimulationWallet
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
        diagnostics: ModelDiagnosticsService,
        position_store: PositionStore,
        wallet: SimulationWallet,
        config: Config,
        set_kalman_enabled: Callable[[bool], None] | None = None,
        get_kalman_enabled: Callable[[], bool] | None = None,
        get_risk_diagnostics: Callable[[], dict] | None = None,
    ) -> None:
        self.performance_tracker = performance_tracker
        self.metrics_agg = metrics_agg
        self.diagnostics = diagnostics
        self.position_store = position_store
        self.wallet = wallet
        self.config = config
        self.set_kalman_enabled = set_kalman_enabled
        self.get_kalman_enabled = get_kalman_enabled
        self.get_risk_diagnostics = get_risk_diagnostics

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
        return self.diagnostics.summary()

    def get_positions(self) -> dict:
        return self.position_store.snapshot()

    def get_health(self) -> HealthStatus:
        return HealthStatus(ok=True, details={"env": self.config.env})

    def get_simulation_results(self) -> dict:
        return {
            "wallet": self.wallet.snapshot(self.position_store),
            "positions": self.position_store.snapshot(),
            "performance": asdict(self.performance_tracker.snapshot()),
            "risk": self.get_risk(),
        }

    def get_timeseries(self, limit: int = 300) -> dict:
        wallet_history = self.wallet.history[-limit:]
        model = self.get_model_summary()
        recent_prob = model.get("recent_prob_up", [])
        if isinstance(recent_prob, list):
            recent_prob = recent_prob[-limit:]
        return {
            "wallet_history": wallet_history,
            "recent_prob_up": recent_prob,
        }

    def get_runtime_config(self) -> dict:
        enabled = bool(self.get_kalman_enabled()) if self.get_kalman_enabled else bool(self.config.model.get("use_kalman", True))
        return {
            "use_kalman": enabled,
        }

    def update_runtime_config(self, use_kalman: bool) -> dict:
        self.config.model["use_kalman"] = bool(use_kalman)
        if self.set_kalman_enabled is not None:
            self.set_kalman_enabled(bool(use_kalman))
        return self.get_runtime_config()


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
        return json.dumps(payload, default=self._json_default).encode("utf-8")

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
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                path = parsed.path
                query = parse_qs(parsed.query)

                if path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(_DASHBOARD_HTML.encode("utf-8"))
                    return

                routes = {
                    "/health": api.get_health,
                    "/performance": api.get_performance,
                    "/metrics": api.get_metrics,
                    "/risk": api.get_risk,
                    "/model": api.get_model_summary,
                    "/positions": api.get_positions,
                    "/simulation": api.get_simulation_results,
                    "/config": api.get_runtime_config,
                }
                if path == "/timeseries":
                    raw_limit = query.get("limit", ["300"])[0]
                    try:
                        limit = max(10, min(2000, int(raw_limit)))
                    except ValueError:
                        limit = 300
                    payload = api.get_timeseries(limit=limit)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(serializer(payload))
                    return

                handler = routes.get(path)
                if handler is None:
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(serializer({"error": "not_found", "path": path}))
                    return

                payload = handler()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(serializer(payload))

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                path = parsed.path
                if path != "/config/use_kalman":
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(serializer({"error": "not_found", "path": path}))
                    return

                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    body = json.loads(raw.decode("utf-8"))
                except Exception:
                    body = {}

                enabled = bool(body.get("enabled", True))
                payload = api.update_runtime_config(enabled)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(serializer(payload))

            def log_message(self, format: str, *args: object) -> None:
                # Csendes HTTP handler, hogy a terminal ne legyen tele loggal.
                _ = format
                _ = args

        self._httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
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


_DASHBOARD_HTML = """<!doctype html>
<html lang="hu">
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
