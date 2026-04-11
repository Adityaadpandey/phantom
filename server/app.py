"""
PHANTOM — OpenEnv Server Entry Point

This module provides the standard OpenEnv server entry point.
It wraps the FastAPI app from phantom.api and starts uvicorn.
"""

from __future__ import annotations

import argparse
import os
import uvicorn

from phantom.api import app


def main() -> None:
    """Start the PHANTOM OpenEnv server."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    port = args.port if args.port is not None else int(os.getenv("API_PORT", "7860"))

    uvicorn.run(
        "server.app:app",
        host=args.host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
