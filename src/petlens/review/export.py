"""Turn my reviews into dataset inputs (Step 9 to Step 2).

- Reviews of Commons photos go to ``data/reviews/labels.parquet`` (committed). The
  dataset build applies them: confirmed or corrected labels get ``label_origin = "me"``,
  "not a cat or dog" and "not sure" photos are excluded completely.
- Animals from my own batches I confirmed or corrected become a ``review`` source: a crop of each
  (from the face-blurred stored frame) is stored like any dataset image and listed in
  ``processed/sources/review/manifest.parquet``. Blind-audit answers are recorded but
  flagged ``audit_only`` so they are never trained on.
- Reviewer-check cards are Oxford test photos and never change any label.
- Only reviewed labels are exported; unreviewed predictions never become labels.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from petlens.config import Settings
from petlens.data.catalog import REPO_DATA_DIR
from petlens.data.imaging import ImageRejectedError, store_image, stored_path
from petlens.data.manifest import ManifestRow, write_manifest
from petlens.db import models as m
from petlens.review.cards import NOT_PET

SOURCE_KEY = "review"
LABELS_SCHEMA = pa.schema(
    [
        ("image_id", pa.string()),
        ("species", pa.string()),
        ("label", pa.string()),
        ("action", pa.string()),
        ("queue", pa.string()),
        ("reviewed_at", pa.string()),
    ]
)
CROP_MARGIN = 0.25


def labels_path(repo_data_dir: Path = REPO_DATA_DIR) -> Path:
    return repo_data_dir / "reviews" / "labels.parquet"


def read_labels(path: Path) -> dict[str, dict[str, Any]]:
    """image_id -> latest review, as written by ``export``."""
    if not path.exists():
        return {}
    return {r["image_id"]: r for r in pq.read_table(path, schema=LABELS_SCHEMA).to_pylist()}


def latest_answers(session: Session) -> list[tuple[m.ReviewCard, m.LabelEvent]]:
    """The last answer for every reviewed card (answers are append-only)."""
    pairs = session.execute(
        select(m.ReviewCard, m.LabelEvent)
        .join(m.LabelEvent, m.LabelEvent.review_card_id == m.ReviewCard.id)
        .where(m.ReviewCard.queue != "check", m.LabelEvent.origin == "me")
        .order_by(m.LabelEvent.id)
    ).all()
    latest: dict[Any, tuple[m.ReviewCard, m.LabelEvent]] = {}
    for card, event in pairs:
        latest[card.id] = (card, event)
    return list(latest.values())


def _crop(frame: Path, box: tuple[float, float, float, float]) -> bytes:
    with Image.open(frame) as im:
        img = im.convert("RGB")
    w, h = img.size
    x0, y0, x1, y1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    mx, my = (x1 - x0) * CROP_MARGIN, (y1 - y0) * CROP_MARGIN
    crop = img.crop(
        (max(0, int(x0 - mx)), max(0, int(y0 - my)), min(w, int(x1 + mx)), min(h, int(y1 + my)))
    )
    buf = io.BytesIO()
    crop.save(buf, "PNG")
    return buf.getvalue()


def own_row(card: m.ReviewCard, event: m.LabelEvent, settings: Settings) -> ManifestRow | None:
    if card.image_id is None or None in (card.x0, card.y0, card.x1, card.y1):
        return None
    frame = stored_path(settings.uploads_dir / "frames", card.image_id)
    if not frame.exists():
        return None
    box = (float(card.x0 or 0), float(card.y0 or 0), float(card.x1 or 1), float(card.y1 or 1))
    try:
        stored = store_image(_crop(frame, box), settings.processed_dir / "images")
    except ImageRejectedError:
        return None
    flags = sorted({*stored.flags, *(["audit_only"] if card.queue == "audit" else [])})
    return ManifestRow(
        image_id=stored.image_id,
        source=SOURCE_KEY,
        source_ref=f"{card.upload_id}/{card.detection_id}",
        licence="own-photo",
        attribution="Mohamed",
        species=card.species,
        breed=event.proposed_label,
        label_origin="me",
        width=stored.original_width,
        height=stored.original_height,
        author=f"upload {card.upload_id}",  # groups all animals of one upload together
        video_id=str(card.upload_id),
        flags=flags,
    )


def export(session: Session, settings: Settings, repo_data_dir: Path = REPO_DATA_DIR) -> str:
    commons: list[dict[str, Any]] = []
    own: list[ManifestRow] = []
    skipped = 0
    for card, event in latest_answers(session):
        if card.queue == "commons" and card.image_id:
            commons.append(
                {
                    "image_id": card.image_id,
                    "species": card.species,
                    "label": event.proposed_label,
                    "action": event.action,
                    "queue": card.queue,
                    "reviewed_at": event.created_at.isoformat(),
                }
            )
        elif event.action in ("accept", "fix") and event.proposed_label != NOT_PET:
            row = own_row(card, event, settings)
            if row is None:
                skipped += 1
            else:
                own.append(row)
    path = labels_path(repo_data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    commons.sort(key=lambda r: r["image_id"])
    pq.write_table(pa.Table.from_pylist(commons, schema=LABELS_SCHEMA), path)
    write_manifest(own, settings.processed_dir / "sources" / SOURCE_KEY / "manifest.parquet")
    note = f", {skipped} skipped (frame missing)" if skipped else ""
    return (
        f"exported {len(commons)} Commons reviews to {path} and {len(own)} of my own "
        f"photos as the '{SOURCE_KEY}' source{note}. Run `make data-build` next."
    )
