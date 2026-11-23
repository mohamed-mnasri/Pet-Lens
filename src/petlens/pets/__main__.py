"""``python -m petlens.pets``: pet re-identification report (``make pets-eval``) and a
one-off ``match`` that gives a pet to every stored animal that has none yet."""

from __future__ import annotations

import argparse

from sqlalchemy import select

from petlens.data.catalog import REPO_DATA_DIR
from petlens.db import models as m
from petlens.db.session import session_scope
from petlens.pets import evaluate, match


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m petlens.pets")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("match", help="match stored animals that have no pet yet")
    sub.add_parser("eval", help="re-identification precision and recall")
    args = parser.parse_args(argv)
    with session_scope() as session:
        if args.cmd == "match":
            uploads = list(session.scalars(select(m.Upload.id).order_by(m.Upload.created_at)))
            n = sum(len(match.assign_upload(session, u)) for u in uploads)
            print(f"matched {n} animals in {len(uploads)} uploads")
            return
        text = evaluate.report(session)
    out = REPO_DATA_DIR.parent / "docs" / "results" / "pets-reid.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
