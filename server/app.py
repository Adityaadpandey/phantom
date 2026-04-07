"""
PHANTOM — OpenEnv Server Entry Point

This module provides the standard OpenEnv server entry point.
It wraps the FastAPI app from phantom.api and starts uvicorn.
"""

from __future__ import annotations

import uvicorn

from phantom.api import app  # noqa: F401 — re-export for uvicorn


def main() -> None:
    """Start the PHANTOM OpenEnv server."""
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )


if __name__ == "__main__":
    main()
