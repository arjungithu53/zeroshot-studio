"""
CORS middleware configuration helper.

Usage:
    from backend.shared.utils.cors_config import add_cors_middleware
    add_cors_middleware(app)
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def add_cors_middleware(app: FastAPI) -> None:
    """
    Attach CORSMiddleware to a FastAPI app using environment configuration.

    Reads CORS_ALLOW_ORIGINS (comma-separated) from environment.
    Falls back to allowing localhost:3000 and localhost:8080.
    In production (ENVIRONMENT=production) the PRODUCTION_DOMAIN env var is
    also included when HTTPS_ENABLED=true.
    """
    raw = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000,http://localhost:8080")
    origins = [o.strip() for o in raw.split(",") if o.strip()]

    environment = os.getenv("ENVIRONMENT", "development")
    if environment == "production":
        domain = os.getenv("PRODUCTION_DOMAIN", "")
        https_enabled = os.getenv("HTTPS_ENABLED", "true").lower() == "true"
        if domain:
            scheme = "https" if https_enabled else "http"
            prod_origin = f"{scheme}://{domain}"
            if prod_origin not in origins:
                origins.append(prod_origin)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
