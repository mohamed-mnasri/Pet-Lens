"""Pure helpers for the pages: upload pre-checks, wording and drawing. No Streamlit here,
so they are easy to test."""

from __future__ import annotations

import io
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from petlens.api.ingest import sniff

BADGES = {
    "accept": ("Breed", "green"),
    "no_clear_breed": ("No clear breed", "orange"),
    "review": ("Not sure", "gray"),
}


def precheck(name: str, data: bytes, limits: dict[str, float]) -> tuple[str, str | None]:
    """Return (kind, problem) before anything is sent. ``problem`` is None when it's fine.

    The API checks everything again; this just saves an upload that would be refused.
    """
    ft = sniff(data[:32])
    if ft is None:
        return "unknown", f"{name}: not a JPEG, PNG, WebP, MP4, MOV or WebM file"
    limit = limits["photo_max_bytes"] if ft.kind == "photo" else limits["video_max_bytes"]
    if len(data) > limit:
        return ft.kind, f"{name}: over the {limit / 2**20:.0f} MB {ft.kind} limit"
    if ft.kind == "photo":
        try:
            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
        except Exception:
            return ft.kind, f"{name}: the photo can't be read"
        if w * h > limits["photo_max_pixels"]:
            mp = limits["photo_max_pixels"] / 1e6
            return ft.kind, f"{name}: {w * h / 1e6:.0f} megapixels, over the {mp:.0f} MP limit"
    return ft.kind, None


def nice(label: str | None) -> str:
    return (label or "").replace("_", " ").title()


def headline(p: dict[str, Any]) -> str:
    """One line for a decision, in the words the page shows."""
    if p["outcome"] == "accept":
        also = f" (or {nice(p['also'])})" if p.get("also") else ""
        return f"Looks like a {nice(p['label'])}{also}, {p['confidence']:.0%}"
    if p["outcome"] == "no_clear_breed":
        return "Domestic cat, no clear breed" if p["species"] == "cat" else "No clear breed"
    return "Not sure: sent to review"


def badge(p: dict[str, Any]) -> tuple[str, str]:
    return BADGES[p["outcome"]]


def limits_text(limits: dict[str, float]) -> str:
    return (
        f"Photos: JPEG, PNG or WebP, up to {limits['photo_max_bytes'] / 2**20:.0f} MB and "
        f"{limits['photo_max_pixels'] / 1e6:.0f} MP. Clips: MP4, MOV or WebM, up to "
        f"{limits['video_max_bytes'] / 2**20:.0f} MB and {limits['video_max_seconds']:.0f} s."
    )


def draw_boxes(frame: bytes, detections: list[dict[str, Any]], labels: dict[str, str]) -> bytes:
    """Draw each detection's box (normalised coordinates) with its label on the frame."""
    with Image.open(io.BytesIO(frame)) as im:
        img = im.convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    width = max(2, round(max(w, h) / 300))
    font = ImageFont.load_default(size=max(12, round(max(w, h) / 40)))
    for d in detections:
        box = (d["x0"] * w, d["y0"] * h, d["x1"] * w, d["y1"] * h)
        draw.rectangle(box, outline=(255, 140, 0), width=width)
        text = labels.get(d["id"], d["species"])
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tx, ty = box[0], max(0.0, box[1] - (bottom - top) - 2 * width)
        draw.rectangle(
            (tx, ty, tx + right - left + 2 * width, ty + bottom + width), fill=(255, 140, 0)
        )
        draw.text((tx + width, ty), text, fill=(0, 0, 0), font=font)
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


def detection_labels(run: dict[str, Any]) -> dict[str, str]:
    """Short label per detection: its own answer, or its clip group's answer."""
    by_det = {p["detection_id"]: p for p in run["predictions"] if p.get("detection_id")}
    by_group = {p["group_idx"]: p for p in run["predictions"] if p.get("group_idx") is not None}
    out = {}
    for d in run["detections"]:
        p = by_det.get(d["id"]) or by_group.get(d.get("group_idx"))
        if p is None:
            out[d["id"]] = d["species"]
        elif p["outcome"] == "accept":
            out[d["id"]] = nice(p["label"])
        else:
            out[d["id"]] = badge(p)[0].lower()
        if d.get("group_idx") is not None:
            out[d["id"]] = f"#{d['group_idx'] + 1} {out[d['id']]}"
    return out


def job_progress(job: dict[str, Any]) -> float:
    total = job.get("progress_total") or 0
    return (
        min(1.0, job["progress_done"] / total) if total else (1.0 if job["state"] == "done" else 0)
    )


QUEUE_NAMES = {
    "commons": "Commons",
    "batch": "My batches",
    "audit": "Blind audit",
    "check": "Check",
}


def stats_text(stats: dict[str, Any]) -> str:
    pending = stats.get("pending", {})
    left = sum(pending.values())
    parts = [f"{left} cards left"]
    parts += [f"{QUEUE_NAMES.get(q, q)} {n}" for q, n in sorted(pending.items()) if q != "check"]
    parts.append(f"{stats.get('answered', 0)} answered")
    acc = stats.get("reviewer_accuracy")
    if acc is not None:
        parts.append(
            f"your accuracy on check cards {acc:.0%} "
            f"({stats['checks_correct']}/{stats['checks_answered']})"
        )
    return " · ".join(parts)


def option_label(label: str) -> str:
    if label == "not_pet":
        return "Not a cat or dog"
    if label == "domestic_cat":
        return "Domestic / no clear breed"
    return nice(label)
