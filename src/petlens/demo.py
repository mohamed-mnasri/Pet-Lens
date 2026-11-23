"""One-command demo (Step 17): ``make up && make demo`` from a fresh clone.

Creates a separate ``petlens_demo`` database next to the real one (never touches it),
fills it with the CI fixtures plus two demo pets, builds the dbt marts, then starts the
API on port 8100 and the page on port 8601 and opens the browser. No dataset or trained
model is needed: the tabs that only read the database work; identifying a photo answers
"no trained model on this machine" until ``make train`` has run.

``python -m petlens.demo``       seed and serve (Ctrl+C stops both)
``python -m petlens.demo seed``  only seed the demo database
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import io
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import httpx
from fastapi import FastAPI
from PIL import Image, ImageDraw
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from petlens.config import get_settings
from petlens.data.imaging import stored_path
from petlens.db import models as m

DEMO_DB = "petlens_demo"
DEMO_DIR = Path(".petlens-demo")
API_PORT = 8100
UI_PORT = 8601
HEADLESS = ["--server.headless", "true"]
UI_APP = Path(__file__).resolve().parent / "ui" / "app.py"

# Two made-up pets, so the Pets tab has something to show: (name, species, breed, hue).
PETS = [("Milo", "cat", "ragdoll", 0.08), ("Luna", "dog", "beagle", 0.6), (None, "cat", None, 0.0)]


def demo_url(base_url: str) -> str:
    return make_url(base_url).set(database=DEMO_DB).render_as_string(hide_password=False)


def ensure_database(base_url: str) -> str:
    """Create the demo database on the same server if it isn't there yet."""
    url = demo_url(base_url)
    engine = create_engine(base_url, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        exists = conn.scalar(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": DEMO_DB})
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{DEMO_DB}"'))
    engine.dispose()
    return url


def crop_image(hue: float, seed: int) -> bytes:
    """A small flat drawing of an animal face in the pet's colour (no real photos)."""
    r, g, b = (int(255 * c) for c in colorsys.hsv_to_rgb(hue, 0.45, 0.55 + 0.08 * (seed % 4)))
    img = Image.new("RGB", (192, 192), (238, 232, 222))
    d = ImageDraw.Draw(img)
    shift = 6 * (seed % 3) - 6
    d.polygon([(40 + shift, 70), (60 + shift, 20), (85 + shift, 62)], fill=(r, g, b))
    d.polygon([(107 + shift, 62), (132 + shift, 20), (152 + shift, 70)], fill=(r, g, b))
    d.ellipse((36 + shift, 46, 156 + shift, 166), fill=(r, g, b))
    for x in (72, 120):
        d.ellipse((x - 9 + shift, 92, x + 9 + shift, 110), fill=(30, 26, 22))
    d.ellipse((88 + shift, 122, 104 + shift, 134), fill=(60, 40, 36))
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=80)
    return buf.getvalue()


def seed_pets(url: str, data_dir: Path) -> int:
    """Add the demo pets: uploads, frames, detections with crops, predictions, events."""
    crops_dir = data_dir / "uploads" / "crops"
    engine = create_engine(url)
    added = 0
    with Session(engine) as s:
        for p, (name, species, breed, hue) in enumerate(PETS):
            pet = m.Pet(species=species, name=name, status="named" if name else "candidate")
            s.add(pet)
            s.flush()
            for i in range(4 if name else 2):
                data = crop_image(hue, i)
                sha = hashlib.sha256(data).hexdigest()
                path = stored_path(crops_dir, sha)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                up = m.Upload(
                    kind="photo",
                    original_name=f"demo-{p}-{i}.jpg",
                    mime="image/jpeg",
                    size_bytes=len(data),
                    sha256=hashlib.sha256(f"demo-upload-{p}-{i}".encode()).hexdigest(),
                    status="done",
                )
                s.add(up)
                s.flush()
                frame = m.Frame(
                    upload_id=up.id,
                    frame_idx=0,
                    image_id=sha,
                    path=str(path),
                    width=192,
                    height=192,
                    size_bytes=len(data),
                )
                s.add(frame)
                s.flush()
                det = m.Detection(
                    frame_id=frame.id,
                    species=species,
                    score=0.95,
                    x0=0.1,
                    y0=0.1,
                    x1=0.9,
                    y1=0.9,
                    group_idx=0,
                    crop_path=str(path),
                    pet_id=pet.id,
                )
                s.add(det)
                s.flush()
                top = breed or ("domestic_cat" if species == "cat" else "mixed")
                s.add(
                    m.Prediction(
                        upload_id=up.id,
                        detection_id=det.id,
                        group_idx=0,
                        model_version="demo",
                        species=species,
                        top3=[{"label": top, "p": 0.8 - 0.05 * i}],
                        knn_distance=0.2,
                        outcome="accept" if breed else "no_clear_breed",
                        confidence=0.8 - 0.05 * i,
                        label=breed,
                    )
                )
                action = "new_pet" if i == 0 else "assign"
                s.add(
                    m.IdentityEvent(
                        pet_id=pet.id, detection_id=det.id, action=action, origin="model"
                    )
                )
                added += 1
            if name:
                s.add(m.IdentityEvent(pet_id=pet.id, action="name", origin="me", value=name))
        s.commit()
    engine.dispose()
    return added


def dbt_build(url: str) -> int:
    from petlens.eval.dbt_run import DBT_DIR, env_for

    try:
        return subprocess.call(["dbt", "build"], cwd=DBT_DIR, env=os.environ | env_for(url))
    except FileNotFoundError:
        print("dbt isn't installed: run `uv sync --group dbt` (make demo does this).")
        return 1


def seed(base_url: str, data_dir: Path = DEMO_DIR) -> str:
    from petlens.eval import fixtures

    url = ensure_database(base_url)
    fixtures.main(url)  # resets the demo database only
    print(f"demo: {seed_pets(url, data_dir)} pet photos in {data_dir}")
    if dbt_build(url) != 0:
        print("demo: dbt build failed; the Model health tab will be empty.")
    return url


def api_app() -> FastAPI:
    """The API without a trained model (uvicorn factory, used by ``serve``)."""
    from petlens.api.app import create_app

    return create_app(require_models=False)


def wait_for(url: str, seconds: float = 60) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            if httpx.get(url, timeout=2).status_code < 500:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    return False


def serve(url: str, data_dir: Path = DEMO_DIR, open_browser: bool = True) -> int:
    env = os.environ | {
        "DATABASE_URL": url,
        "DATA_DIR": str(data_dir.resolve()),
        "PETLENS_API_URL": f"http://127.0.0.1:{API_PORT}",
    }
    py = sys.executable
    api = subprocess.Popen(
        [py, "-m", "uvicorn", "petlens.demo:api_app", "--factory", "--port", str(API_PORT)],
        env=env,
    )
    ui = subprocess.Popen(
        [py, "-m", "streamlit", "run", str(UI_APP), "--server.port", str(UI_PORT), *HEADLESS],
        env=env,
    )
    page = f"http://127.0.0.1:{UI_PORT}"
    try:
        if wait_for(f"http://127.0.0.1:{API_PORT}/health") and wait_for(page):
            print(f"\nPetLens demo: {page}  (live camera: http://127.0.0.1:{API_PORT}/live)")
            print("Ctrl+C stops it.\n")
            if open_browser:
                webbrowser.open(page)
        while api.poll() is None and ui.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in (ui, api):
            proc.terminate()
        for proc in (ui, api):
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m petlens.demo")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "seed"])
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    url = seed(get_settings().database_url)
    if args.command == "seed":
        return 0
    return serve(url, open_browser=not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
