"""Command line entry point: ``python -m petlens.data <stage>``."""

from __future__ import annotations

import argparse
import logging
import os

from petlens.config import get_settings
from petlens.data import oxford
from petlens.data.catalog import load_breeds, load_sources


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m petlens.data")
    sub = parser.add_subparsers(dest="stage", required=True)
    ox = sub.add_parser("oxford", help="fetch and ingest Oxford-IIIT Pet")
    ox.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    breeds = load_breeds()
    sources = load_sources()

    if args.stage == "oxford":
        path = oxford.build(
            settings.data_dir, breeds, sources.sources[oxford.SOURCE_KEY], workers=args.workers
        )
        print(f"manifest: {path}")


if __name__ == "__main__":
    main()
