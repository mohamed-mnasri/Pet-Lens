"""Database migrations: ``python -m petlens.db upgrade|downgrade|revision``."""

from __future__ import annotations

import argparse

from petlens.db import migrate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m petlens.db")
    sub = parser.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("upgrade", help="apply migrations")
    up.add_argument("revision", nargs="?", default="head")
    down = sub.add_parser("downgrade", help="undo migrations")
    down.add_argument("revision", nargs="?", default="-1")
    rev = sub.add_parser("revision", help="autogenerate a migration from the models")
    rev.add_argument("message")
    sub.add_parser("index", help="load training embeddings into train_image (evidence panel)")
    args = parser.parse_args(argv)

    if args.cmd == "upgrade":
        migrate.upgrade(revision=args.revision)
    elif args.cmd == "downgrade":
        migrate.downgrade(revision=args.revision)
    elif args.cmd == "index":
        from petlens.config import get_settings
        from petlens.db.evidence import index_from_files
        from petlens.db.session import session_scope

        with session_scope() as session:
            n = index_from_files(session, get_settings())
        print(f"indexed {n} training images for the evidence panel")
    else:
        migrate.revision(args.message)


if __name__ == "__main__":
    main()
