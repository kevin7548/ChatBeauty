"""
HTTP middleware that measures and logs total request latency.

Adds X-Total-Latency-Ms header to every response.
"""

import time
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger(__name__)


class LatencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        total_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Total-Latency-Ms"] = f"{total_ms:.1f}"
        logger.info(f"{request.method} {request.url.path} completed in {total_ms:.1f}ms")
        return response
