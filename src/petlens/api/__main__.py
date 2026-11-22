"""Run the API locally: ``python -m petlens.api`` (binds to 127.0.0.1 by default)."""

from __future__ import annotations

import uvicorn

from petlens.api.app import create_app
from petlens.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(create_app(settings), host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
