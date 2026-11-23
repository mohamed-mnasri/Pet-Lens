"""The Model Health page's data. It reads only the dbt marts (``make dbt`` builds them),
never the raw tables, so the page shows exactly what the analytics layer says."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

QUERIES = {
    "registry": """
        select run_id, dataset_version, embedding_model, is_champion, registered_at,
               promoted_at, gate_passed, gate_golden_gain, gate_unseen_gain,
               golden_top1, unseen_top1, false_breed_claim_rate
        from marts.mart_model_registry order by registered_at desc limit 10""",
    "accuracy": """
        select species, test_set, n, top1, top1_lo, top1_hi, top3, macro_f1, ece, coverage,
               selective_accuracy, false_breed_claim_rate, unknown_rejection_rate, run_id,
               evaluated_at
        from marts.mart_accuracy
        where eval_run_id = (
            select eval_run_id from marts.mart_accuracy order by evaluated_at desc limit 1)
        order by species, test_set""",
    "confused": """
        select species, test_set, true_label, predicted_label, n_confused, n_label,
               share_of_label
        from marts.mart_confused_pairs
        where eval_run_id = (
            select eval_run_id from marts.mart_accuracy order by evaluated_at desc limit 1)
          and test_set in ('test_unseen_source', 'golden')
        order by n_confused desc, share_of_label desc limit 10""",
    "abstention": """
        select day, species, n, accept_rate, no_clear_breed_rate, review_rate
        from marts.mart_abstention where scope = 'live'
        order by day desc limit 14""",
    "reviews": """
        select day, answers, accepted, fixed, rejected, unsure, skipped, median_review_ms,
               checks_answered, checks_right
        from marts.mart_review_throughput order by day desc limit 14""",
}


def _rows(session: Session, sql: str) -> list[dict[str, Any]]:
    return [dict(r._mapping) for r in session.execute(text(sql))]


def model_health(session: Session) -> dict[str, Any]:
    out: dict[str, Any] = {"ready": True}
    for key, sql in QUERIES.items():
        try:
            with session.begin_nested():
                out[key] = _rows(session, sql)
        except ProgrammingError:
            out[key] = []
            out["ready"] = False  # a mart is missing: `make dbt` hasn't run yet
    return out
