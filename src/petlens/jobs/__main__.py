"""Jobs from the terminal.

python -m petlens.jobs worker                 # run the worker (make worker)
python -m petlens.jobs review-batch FOLDER    # queue a folder of photos and clips
python -m petlens.jobs screen                 # queue Commons screening
python -m petlens.jobs status                 # recent jobs and their progress
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from petlens.config import get_settings
from petlens.db import models as m
from petlens.db.session import get_engine
from petlens.jobs import queue
from petlens.jobs.handlers import batch_files
from petlens.jobs.worker import Worker


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m petlens.jobs")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("worker", help="run the worker until Ctrl+C")
    batch = sub.add_parser("review-batch", help="queue a folder of photos and clips")
    batch.add_argument("folder", type=Path)
    sub.add_parser("screen", help="queue Commons screening")
    sub.add_parser("status", help="show recent jobs")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = get_settings()
    if args.cmd == "worker":
        worker = Worker(settings)
        worker.install_signal_handlers()
        print("worker started; Ctrl+C stops it after the current job")
        worker.run()
        return

    sessions = sessionmaker(get_engine(settings.database_url), expire_on_commit=False)
    with sessions() as session:
        if args.cmd == "review-batch":
            folder = args.folder.resolve()
            files = batch_files(folder)  # fail now, not in the worker, if the folder is bad
            job = queue.enqueue(session, "review_batch", {"folder": str(folder)}, total=len(files))
            session.commit()
            print(f"queued job {job.id}: {len(files)} files from {folder}")
        elif args.cmd == "screen":
            job = queue.enqueue(session, "commons_screening", max_attempts=1)
            session.commit()
            print(f"queued job {job.id}")
        else:
            jobs = session.scalars(select(m.Job).order_by(m.Job.created_at.desc()).limit(20))
            for j in jobs:
                total = j.progress_total if j.progress_total is not None else "?"
                err = f"  error: {j.error.splitlines()[0]}" if j.error else ""
                print(
                    f"{j.id}  {j.kind:<18} {j.state:<8} {j.progress_done}/{total}"
                    f"  attempt {j.attempts}/{j.max_attempts}{err}"
                )


if __name__ == "__main__":
    main()
