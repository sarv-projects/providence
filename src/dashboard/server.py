"""
Zero-dependency web dashboard for the LLM gateway.

Runs on the Python standard library (``ThreadingHTTPServer`` + SSE), so it needs
no web framework. It reads the shared process-wide ``DEFAULT_METRICS`` registry
that the gateway writes into, and live-streams events to the browser.

Endpoints:
    /                  -> the dashboard SPA (src/dashboard/static_index.html)
    /api/status        -> JSON snapshot of metrics + circuits + rate limits
    /api/events        -> Server-Sent Events stream of gateway events
    /metrics           -> Prometheus text format (for scraping / Grafana)

Run:
    uv run python -m src.dashboard [--port 8080]
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..gateway.metrics import DEFAULT_METRICS

STATIC_FILE = Path(__file__).parent / "static_index.html"

# Store a monotonic cursor so SSE only sends *new* events per connection.
_last_event_ts = [0.0]


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter logs
        pass

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_static()
        elif self.path == "/api/status":
            snap = DEFAULT_METRICS.snapshot()
            snap["generated_at"] = time.time()
            # Search-cache observability from the tool bus (size, TTL, hit rate).
            try:
                from src.tools.registry import get_registry
                snap["search_cache"] = get_registry().cache_stats()
            except Exception:
                snap["search_cache"] = {}
            self._send_json(snap)
        elif self.path == "/api/events":
            self._serve_sse()
        elif self.path == "/api/research/progress":
            self._serve_research_sse()
        elif self.path == "/metrics":
            body = DEFAULT_METRICS.to_prometheus().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")

    def _serve_static(self) -> None:
        if not STATIC_FILE.exists():
            self.send_response(404)
            self.end_headers()
            return
        body = STATIC_FILE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self) -> None:
        """Stream gateway events to the browser as Server-Sent Events."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            # Bound the stream: an undisconnect-detected client otherwise
            # leaks a ThreadingHTTPServer thread forever (no finished flag).
            for _ in range(600):  # ~15 min at 1.5s
                snap = DEFAULT_METRICS.snapshot()
                events = [e for e in snap.get("event_log", []) if e.get("ts", 0) > _last_event_ts[0]]
                for e in events[-10:]:
                    payload = f"data: {json.dumps(e)}\n\n"
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
                if events:
                    _last_event_ts[0] = max(x.get("ts", 0) for x in events)
                time.sleep(1.5)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_research_sse(self) -> None:
        """Stream research progress to the browser as Server-Sent Events."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            from src.engine.progress import get_progress
            progress = get_progress()
            while True:
                snap = progress.snapshot()
                payload = f"data: {json.dumps(snap)}\n\n"
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()
                if snap.get("finished"):
                    # Send final snapshot then close
                    time.sleep(0.5)
                    break
                time.sleep(1.0)
        except (BrokenPipeError, ConnectionResetError):
            pass


def serve(port: int = 8080) -> None:
    httpd = ThreadingHTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"\n  Dashboard: http://localhost:{port}")
    print(f"  Prometheus metrics: http://localhost:{port}/metrics\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM gateway dashboard")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    serve(args.port)


if __name__ == "__main__":
    main()
