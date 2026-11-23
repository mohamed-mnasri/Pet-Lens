"""``python -m petlens.vlm``: the VLM screener (Step 13) and breed experiment (Step 14).

screen [--limit N]   queue a vlm_screen job for the worker (make vlm-screen)
screen --now         run it here instead, without the worker
export               write the tags to data/vlm/tags.parquet (make vlm-export)
report               agreement and effect report (make vlm-report)
breeds [--limit N]   the one-off breed-naming experiment (make vlm-breeds)
"""

from __future__ import annotations

import argparse
import logging

from petlens.config import get_settings
from petlens.data.catalog import REPO_DATA_DIR
from petlens.data.manifest import read_manifest
from petlens.data.vlm_flags import TAGS_PATH
from petlens.db.session import session_scope

RESULTS = REPO_DATA_DIR.parent / "docs" / "results"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m petlens.vlm")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen", help="tag photos with the local VLM")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--now", action="store_true", help="run here instead of queueing a job")
    sub.add_parser("export", help="write the tags for make data-build")
    sub.add_parser("report", help="agreement with my reviews and the photo checks")
    b = sub.add_parser("breeds", help="zero-shot breed naming vs the classifier (Step 14)")
    b.add_argument("--limit", type=int, default=None, help="photos per test set")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    version = (REPO_DATA_DIR / "manifests" / "LATEST").read_text(encoding="utf-8").strip()
    rows = read_manifest(REPO_DATA_DIR / "manifests" / f"{version}.parquet")

    from petlens.vlm import screen

    if args.cmd == "breeds":
        from petlens.vlm import breeds

        out = RESULTS / f"vlm-breeds-{version}.md"
        text = breeds.run_experiment(settings, rows, args.limit)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(text)
        print(f"written to {out}")
        return
    with session_scope() as session:
        if args.cmd == "screen" and not args.now:
            from petlens.jobs import queue

            job = queue.enqueue(session, "vlm_screen", {"limit": args.limit}, max_attempts=3)
            print(f"queued job {job.id}; `make worker` runs it (stop the camera page first)")
        elif args.cmd == "screen":
            vlm = screen.OllamaVlm(settings.vlm_model, settings.ollama_url)
            counts = screen.run(
                session,
                rows,
                settings.data_dir,
                vlm,
                args.limit,
                progress=lambda d, t: print(f"\r{d}/{t}", end="", flush=True),
            )
            print(f"\n{counts}")
        elif args.cmd == "export":
            n = screen.export(session, settings.vlm_model)
            print(f"{n} tags written to {TAGS_PATH}. Next: make data-build")
        else:
            from petlens.vlm import measure

            text = measure.report(session, rows, settings.vlm_model)
            out = RESULTS / f"vlm-screen-{screen.PROMPT_VERSION}.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(text)
            print(f"written to {out}")


if __name__ == "__main__":
    main()
