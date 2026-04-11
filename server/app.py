"""
PHANTOM — OpenEnv Server Entry Point

This module provides the standard OpenEnv server entry point.
It wraps the FastAPI app from phantom.api and starts uvicorn.
"""

from __future__ import annotations

import argparse
import os
import uvicorn

from phantom.api import app  # noqa: F401 — re-export for uvicorn


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    """Start the PHANTOM OpenEnv server."""
    if port is None:
        port = int(os.getenv("API_PORT", "7860"))

    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
