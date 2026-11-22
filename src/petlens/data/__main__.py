"""Command line entry point: ``python -m petlens.data <stage>``."""

from __future__ import annotations

import argparse
import logging
import os

import truststore

from petlens.config import get_settings
from petlens.data import commons, oxford
from petlens.data.catalog import load_breeds, load_sources


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m petlens.data")
    sub = parser.add_subparsers(dest="stage", required=True)
    ox = sub.add_parser("oxford", help="fetch and ingest Oxford-IIIT Pet")
    ox.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    co = sub.add_parser("commons", help="fetch and ingest Wikimedia Commons cat photos")
    co.add_argument(
        "--max-per-category",
        type=int,
        default=None,
        help="keep at most this many files per label (handy for a quick first run)",
    )
    co.add_argument("--depth", type=int, default=1, help="how deep to follow subcategories")
    args = parser.parse_args(argv)
    # Verify HTTPS with the OS trust store (fixes stale intermediates on Windows).
    truststore.inject_into_ssl()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    breeds = load_breeds()
    sources = load_sources()

    if args.stage == "oxford":
        path = oxford.build(
            settings.data_dir, breeds, sources.sources[oxford.SOURCE_KEY], workers=args.workers
        )
        print(f"manifest: {path}")
    elif args.stage == "commons":
        path = commons.build(
            settings.data_dir,
            breeds,
            sources.sources[commons.SOURCE_KEY],
            max_per_category=args.max_per_category,
            max_depth=args.depth,
        )
        print(f"manifest: {path}")


if __name__ == "__main__":
    main()
