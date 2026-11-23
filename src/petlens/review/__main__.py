"""Review queues from the terminal.

python -m petlens.review load     # Commons and reviewer-check cards (make review-load)
python -m petlens.review export   # reviewed labels for the next dataset (make review-export)
python -m petlens.review stats    # what's left, and my accuracy on check cards
"""

from __future__ import annotations

import argparse

from sqlalchemy import func, select

from petlens.config import get_settings
from petlens.data.catalog import REPO_DATA_DIR, load_breeds
from petlens.data.manifest import read_manifest
from petlens.db import models as m
from petlens.db.session import session_scope
from petlens.models import screening
from petlens.review import answers, cards


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m petlens.review")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("load", help="queue Commons review cards and reviewer-check cards")
    sub.add_parser("export", help="write reviewed labels for the next dataset version")
    sub.add_parser("stats", help="show queues and reviewer accuracy")
    args = parser.parse_args(argv)
    settings = get_settings()

    with session_scope() as session:
        if args.cmd == "load":
            version = (REPO_DATA_DIR / "manifests" / "LATEST").read_text(encoding="utf-8").strip()
            rows = read_manifest(REPO_DATA_DIR / "manifests" / f"{version}.parquet")
            results = screening.read(REPO_DATA_DIR / "screening" / f"{version}.parquet")
            dropped = cards.drop_weak_cards(session, rows)
            new = cards.insert_cards(session, cards.commons_cards(rows, results))
            total = int(
                session.execute(select(func.count()).select_from(m.ReviewCard)).scalar_one()
            )
            n_checks = max(1, round(total * cards.CHECK_RATE / (1 - cards.CHECK_RATE)))
            checks = cards.insert_cards(session, cards.check_cards(rows, load_breeds(), n_checks))
            print(f"queued {new} Commons cards and {checks} reviewer-check cards ({version})")
            if dropped:
                print(f"took {dropped} weak photos off the queue (black and white, blurry, ...)")
        elif args.cmd == "export":
            from petlens.review.export import export

            summary = export(session, settings)
            print(summary)
        else:
            s = answers.stats(session)
            print(f"pending: {s.pending or 'none'}; answered: {s.answered}")
            if s.reviewer_accuracy is not None:
                print(
                    f"reviewer check: {s.checks_correct}/{s.checks_answered} right "
                    f"({s.reviewer_accuracy:.0%})"
                )


if __name__ == "__main__":
    main()
