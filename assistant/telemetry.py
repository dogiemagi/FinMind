"""Request telemetry + a simple per-IP rate limiter (Phase 11: monitoring)."""
from __future__ import annotations

import time
from collections import defaultdict, deque

from django.conf import settings
from django.http import HttpResponse

METRICS: dict[str, float] = defaultdict(float)
_HITS: dict[str, deque] = defaultdict(deque)


class TelemetryMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path.startswith("/api/"):
            ip, now = request.META.get("REMOTE_ADDR", "x"), time.time()
            q = _HITS[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= settings.FINMIND_RATE_LIMIT_PER_MIN:
                METRICS["rate_limited_total"] += 1
                return HttpResponse("rate limit exceeded", status=429, content_type="text/plain")
            q.append(now)
        t = time.perf_counter()
        response = self.get_response(request)
        if request.path.startswith("/api/"):
            METRICS["requests_total"] += 1
            METRICS["latency_seconds_sum"] += time.perf_counter() - t
        return response
