"""``make register``, ``make gate``, ``make promote`` and ``make registry``."""

from __future__ import annotations

from petlens.config import get_settings
from petlens.data.catalog import load_breeds
from petlens.db.session import session_scope
from petlens.models import golden, promotion, registry
from petlens.models.runs import find_run, load_run


def run(stage: str, run_id: str | None) -> None:
    settings = get_settings()
    with session_scope() as session:
        if stage == "registry":
            entries = registry.listing(session)
            if not entries:
                print("no registered runs yet; run `make register`")
            for e in entries:
                star = "*" if e.champion else " "
                verdict = (
                    "" if e.gate is None else ("gate passed" if e.gate["passed"] else "gate failed")
                )
                print(f"{star} {e.run_id}  {e.dataset_version}  {verdict}")
            print("(* = champion, served by the app)")
            return
        challenger = load_run(find_run(settings.models_dir, run_id))
        entry = registry.register(session, challenger)
        print(f"registered {challenger.run_id} (weights {entry.weights_hash[:12]})")
        if stage == "register":
            return
        best = registry.champion(session)
        champion = None
        if best is not None and best.run_id != challenger.run_id:
            champion = load_run(find_run(settings.models_dir, best.run_id))
        elif best is not None:
            print("this run is already the champion")
            return
        result = promotion.compare(
            champion, challenger, golden.read(), settings.data_dir, load_breeds()
        )
        report = result.render()
        (challenger.path / "gate.md").write_text(report, encoding="utf-8")
        registry.record_gate(session, entry, result.as_dict(), promote=stage == "promote")
        print(report)
        if stage == "promote" and result.passed:
            print(f"{challenger.run_id} is now the champion. Restart `make api` to serve it.")
        elif stage == "gate" and result.passed:
            print("The gate passes. Run `make promote` to make it the champion.")
