"""Disk usage per data folder: ``python -m petlens.storage`` (``make storage-report``)."""

from __future__ import annotations

from pathlib import Path

from petlens.config import Settings, get_settings


def folder_usage(root: Path) -> tuple[int, int]:
    """Total bytes and file count under ``root`` (0, 0 if it doesn't exist)."""
    if not root.exists():
        return 0, 0
    sizes = [p.stat().st_size for p in root.rglob("*") if p.is_file()]
    return sum(sizes), len(sizes)


def report(settings: Settings) -> str:
    folders = {
        "uploads/frames": settings.uploads_dir / "frames",
        "uploads/crops": settings.uploads_dir / "crops",
        "inbox (should be empty)": settings.inbox_dir,
        "raw": settings.raw_dir,
        "processed": settings.processed_dir,
        "models": settings.models_dir,
        "footage": settings.footage_dir,
    }
    lines = [f"Storage under {settings.data_dir}", ""]
    lines.append(f"{'folder':<26}{'files':>9}{'size':>12}")
    total = 0
    for name, path in folders.items():
        nbytes, count = folder_usage(path)
        total += nbytes
        lines.append(f"{name:<26}{count:>9}{nbytes / 2**20:>9.1f} MB")
    lines.append(f"{'total':<26}{'':>9}{total / 2**20:>9.1f} MB")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(get_settings()))
