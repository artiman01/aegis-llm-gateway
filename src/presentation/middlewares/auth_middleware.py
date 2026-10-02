"""Gateway Authentication and Authorization Middleware.

Enforces Bearer token verification against configured API keys and master keys,
provides path-based exemption for observability and documentation endpoints,
and extracts multi-tenant identity metadata.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from presentation.middlewares.error_handling_middleware import (
    format_openai_error_response,
)

logger = logging.getLogger(__name__)

DEFAULT_PUBLIC_EXEMPTIONS = frozenset(
    {
        "/",
        "/health",
        "/health/live",
        "/health/ready",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
    }
)


class AuthMiddleware(BaseHTTPMiddleware):
    """Enforces Bearer authentication for protected gateway routes."""

    def __init__(
        self,
        app: Any,
        allowed_keys: dict[str, str | None] | set[str] | None = None,
        master_key: str | None = None,
        public_paths: set[str] | frozenset[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._public_paths = (
            frozenset(public_paths) if public_paths is not None else DEFAULT_PUBLIC_EXEMPTIONS
        )
        self._key_tenant_map: dict[str, str | None] = {}

        # 1. Register explicit allowed keys or env-configured keys
        if allowed_keys is not None:
            if isinstance(allowed_keys, (set, frozenset, list)):
                for k in allowed_keys:
                    self._key_tenant_map[str(k).strip()] = None
            elif isinstance(allowed_keys, dict):
                for k, v in allowed_keys.items():
                    self._key_tenant_map[str(k).strip()] = str(v) if v else None
        else:
            self._load_keys_from_env()

        # 2. Register master key fallback
        configured_master = (
            master_key
            if master_key is not None
            else os.getenv("AEGIS_MASTER_KEY", "sk-aegis-master-key")
        )
        if configured_master and configured_master.strip():
            self._key_tenant_map[configured_master.strip()] = "admin"

    def _load_keys_from_env(self) -> None:
        """Parse AEGIS_GATEWAY_KEYS from environment variable (JSON or CSV)."""
        raw_keys = os.getenv("AEGIS_GATEWAY_KEYS", "").strip()
        if not raw_keys:
            return

        # Attempt JSON dictionary parsing
        if raw_keys.startswith("{"):
            try:
                data = json.loads(raw_keys)
                if isinstance(data, dict):
                    for k, val in data.items():
                        tenant: str | None = None
                        if isinstance(val, str):
                            tenant = val
                        elif isinstance(val, dict):
                            tenant = str(val.get("tenant_id") or val.get("tenant", "")) or None
                        self._key_tenant_map[str(k).strip()] = tenant
                    return
            except Exception as exc:
                logger.warning(
                    "Failed parsing AEGIS_GATEWAY_KEYS as JSON (%s); fallback to CSV", exc
                )

        # Fallback to comma-separated list: "key1,key2:tenant2,key3"
        for item in raw_keys.split(","):
            entry = item.strip()
            if not entry:
                continue
            if ":" in entry:
                parts = entry.split(":", 1)
                self._key_tenant_map[parts[0].strip()] = parts[1].strip() or None
            else:
                self._key_tenant_map[entry] = None

    def _is_public(self, path: str) -> bool:
        """Check if request path matches any public exemption."""
        normalized = path.rstrip("/") or "/"
        if normalized in self._public_paths or path in self._public_paths:
            return True
        return any(exempt != "/" and path.startswith(exempt) for exempt in self._public_paths)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Inspect request authentication headers and route identity."""
        # Initialize default state attributes
        request.state.api_key = None
        request.state.tenant_id = None

        # Extract explicit tenant header if provided
        header_tenant = request.headers.get("X-Tenant-ID") or request.headers.get("x-tenant-id")
        if header_tenant:
            request.state.tenant_id = header_tenant.strip()

        # Public path exemption & CORS preflights
        if request.method == "OPTIONS" or self._is_public(request.url.path):
            return await call_next(request)

        # Verify Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.strip():
            return format_openai_error_response(
                status_code=401,
                message="Missing Authorization header. Expected 'Authorization: Bearer <key>'.",
                error_type="authentication_error",
                code="missing_api_key",
            )

        parts = auth_header.split(None, 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return format_openai_error_response(
                status_code=401,
                message="Invalid Authorization header format. Format must be 'Bearer <key>'.",
                error_type="authentication_error",
                code="invalid_authorization_format",
            )

        token = parts[1].strip()
        if token not in self._key_tenant_map:
            return format_openai_error_response(
                status_code=401,
                message="Invalid or unauthorized API key provided.",
                error_type="authentication_error",
                code="invalid_api_key",
            )

        # Extract tenant metadata: Key metadata takes precedence, fallback to header
        key_tenant = self._key_tenant_map.get(token)
        if key_tenant:
            request.state.tenant_id = key_tenant
        elif header_tenant:
            request.state.tenant_id = header_tenant.strip()

        request.state.api_key = token
        return await call_next(request)
