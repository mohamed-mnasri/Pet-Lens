"""PetLens upload page: ``make ui`` (needs ``make api`` running).

Upload photos or short clips, see each animal's decision with its top 3 and the most
similar training photos, and follow batch jobs. The page only talks to the API.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from petlens.ui import client as api_client
from petlens.ui import view

st.set_page_config(page_title="PetLens", page_icon="🐾", layout="wide")


@st.cache_resource
def get_client(base_url: str) -> api_client.ApiClient:
    return api_client.ApiClient(base_url)


def show_prediction(api: api_client.ApiClient, run: dict[str, Any], p: dict[str, Any]) -> None:
    label, colour = view.badge(p)
    title = f"#{p['group_idx'] + 1} " if p.get("group_idx") is not None else ""
    st.markdown(f"{title}:{colour}-badge[{label}] **{view.headline(p)}**")
    for item in p["top3"]:
        st.progress(item["p"], text=f"{view.nice(item['label'])} {item['p']:.0%}")
    st.caption(f"Distance to training photos: {p['knn_distance']:.3f}")
    evidence = run.get("evidence", {}).get(p["id"], [])
    if evidence:
        st.caption(f"Most similar {view.nice(p['label'])} training photos")
        cols = st.columns(len(evidence))
        for col, e in zip(cols, evidence, strict=True):
            with col:
                try:
                    st.image(api.image(e["url"]), width="stretch")
                except api_client.ApiError:
                    st.caption("image missing")
                credit = e.get("attribution") or e["source"]
                st.caption(f"{e['similarity']:.2f} · {credit} · {e.get('licence') or ''}")


def show_run(api: api_client.ApiClient, name: str, run: dict[str, Any]) -> None:
    upload = run["upload"]
    with st.container(border=True):
        st.subheader(name)
        if upload.get("note"):
            st.info(upload["note"])
        labels = view.detection_labels(run)
        frames = run["frames"]
        left, right = st.columns([3, 2])
        with left:
            shown = frames if len(frames) <= 1 else frames[:: max(1, len(frames) // 4)]
            cols = st.columns(len(shown)) if len(shown) > 1 else [st.container()]
            for col, f in zip(cols, shown, strict=True):
                dets = [d for d in run["detections"] if d["frame_id"] == f["id"]]
                frame_bytes = api.image(f"/images/frames/{f['image_id']}")
                with col:
                    st.image(view.draw_boxes(frame_bytes, dets, labels), width="stretch")
                    if upload["kind"] != "photo" or len(frames) > 1:
                        st.caption(f"{f['seconds']:.0f} s")
        with right:
            if not run["predictions"]:
                st.warning("No cat or dog found.")
            for p in run["predictions"]:
                show_prediction(api, run, p)
        faces = sum(f["faces_blurred"] for f in frames)
        if faces:
            st.caption(f"{faces} face(s) blurred in the stored copy.")


def upload_tab(api: api_client.ApiClient, limits: dict[str, float]) -> None:
    st.caption(view.limits_text(limits))
    files = st.file_uploader(
        "Photos or clips",
        type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "webm"],
        accept_multiple_files=True,
    )
    if files and st.button("Analyse", type="primary"):
        for f in files:
            data = f.getvalue()
            kind, problem = view.precheck(f.name, data, limits)
            if problem:
                st.error(problem)
                continue
            with st.spinner(f"Analysing {f.name}…"):
                try:
                    run = api.predict(kind, f.name, data)
                except api_client.ApiError as exc:
                    st.error(f"{f.name}: {exc.detail}")
                    continue
            show_run(api, f.name, run)


def jobs_tab(api: api_client.ApiClient) -> None:
    st.caption("Queue a folder of photos and clips (at most 200). `make worker` runs it.")
    folder = st.text_input("Folder on this PC", placeholder=r"D:\petlens-data\footage\test")
    if st.button("Queue review batch") and folder:
        try:
            job = api.queue_review_batch(folder)
            st.success(f"Queued {job['progress_total']} files.")
        except api_client.ApiError as exc:
            st.error(exc.detail)
    st.button("Refresh")
    jobs = api.jobs()
    if not jobs:
        st.caption("No jobs yet.")
    for j in jobs:
        total = j.get("progress_total")
        text = f"{j['kind']} · {j['state']} · {j['progress_done']}/{total if total else '?'}"
        st.progress(view.job_progress(j), text=text)
        if j.get("error"):
            st.caption(f"Error (attempt {j['attempts']}/{j['max_attempts']}): {j['error'][:200]}")


def main() -> None:
    st.title("🐾 PetLens")
    base_url = os.environ.get("PETLENS_API_URL", api_client.DEFAULT_URL)
    api = get_client(base_url)
    if not api.health():
        st.error(f"The API isn't answering at {base_url}. Start it with `make api`.")
        st.stop()
    limits = api.limits()
    upload, jobs = st.tabs(["Upload", "Batch jobs"])
    with upload:
        upload_tab(api, limits)
    with jobs:
        jobs_tab(api)


main()
