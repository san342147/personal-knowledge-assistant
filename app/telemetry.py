import json
import threading
from collections import deque
from datetime import datetime, timezone
from logging import Formatter, getLogger
from logging.handlers import RotatingFileHandler

from app.config import ROOT


class Telemetry:
    def __init__(self, log_dir=None):
        directory = log_dir or ROOT / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        self.logger = getLogger(f"groundeddesk.{directory}")
        self.logger.setLevel("INFO")
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = RotatingFileHandler(directory / "requests.jsonl", maxBytes=2_000_000,
                                          backupCount=3, encoding="utf-8")
            handler.setFormatter(Formatter("%(message)s"))
            self.logger.addHandler(handler)
        self.lock = threading.Lock()
        self.recent = deque(maxlen=1000)
        self.requests = self.tokens_in = self.tokens_out = 0
        self.cost = 0.0

    def record(self, event):
        event = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        with self.lock:
            self.logger.info(json.dumps(event))
            self.recent.append(event)
            self.requests += 1
            self.tokens_in += event.get("tokens_in", 0)
            self.tokens_out += event.get("tokens_out", 0)
            self.cost += event.get("estimated_cost_usd", 0)

    def snapshot(self):
        with self.lock:
            latencies = sorted(e.get("latency_ms", 0) for e in self.recent)
            return {"scope": "this process; latency window is last 1000 requests",
                "requests": self.requests, "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "estimated_cost_usd": round(self.cost, 8),
                "latency_p50_ms": latencies[len(latencies) // 2] if latencies else None,
                "latency_p95_ms": latencies[min(len(latencies)-1, int(len(latencies)*.95))] if latencies else None}
