"""PetLens page: ``make ui`` (needs ``make api`` running).

Identify photos or short clips (each animal's decision, its top 3 and the most similar
training photos), swipe-review cards, add training photos by breed, and follow batch
jobs. The page only talks to the API.
"""

from __future__ import annotations

import os
import time
from typing import Any

import streamlit as st

from petlens.ui import client as api_client
from petlens.ui import view

st.set_page_config(page_title="PetLens", page_icon="🐾", layout="wide")

# Small layout touches on top of the theme in .streamlit/config.toml.
CSS = """
<style>
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
h1 {font-size: 2rem !important; margin-bottom: 0 !important;}
[data-testid="stCaptionContainer"] {color: #6B6459;}
[data-testid="stVerticalBlockBorderWrapper"] {
  background: #FFFFFF; box-shadow: 0 1px 3px rgba(60, 40, 20, 0.06);
}
[data-testid="stImage"] img {border-radius: 0.6rem;}
.stTabs [data-baseweb="tab-list"] {gap: 0.5rem; border-bottom: 1px solid #E4DACC;}
.stTabs [data-baseweb="tab"] {font-size: 1rem; padding: 0.5rem 0.9rem;}
.pl-headline {font-size: 1.35rem; font-weight: 650; margin: 0.25rem 0 0.5rem;}
.pl-question {font-size: 1.6rem; font-weight: 650; line-height: 1.25; margin: 0 0 0.25rem;}
.pl-muted {color: #6B6459; font-size: 0.9rem;}
footer {visibility: hidden;}
</style>
"""


@st.cache_resource
def get_client(base_url: str) -> api_client.ApiClient:
    return api_client.ApiClient(base_url)


def show_prediction(api: api_client.ApiClient, run: dict[str, Any], p: dict[str, Any]) -> None:
    label, colour = view.badge(p)
    animal = f"Animal #{p['group_idx'] + 1} · " if p.get("group_idx") is not None else ""
    st.markdown(
        f":{colour}-badge[{label}] <span class='pl-muted'>{animal}{p['species']}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='pl-headline'>{view.headline(p)}</div>", unsafe_allow_html=True)
    for item in p["top3"]:
        st.progress(item["p"], text=f"{view.nice(item['label'])} · {item['p']:.0%}")
    evidence = run.get("evidence", {}).get(p["id"], [])
    if evidence:
        with st.expander(f"Why: the most similar {view.nice(p['label'])} training photos"):
            cols = st.columns(len(evidence))
            for col, e in zip(cols, evidence, strict=True):
                with col:
                    try:
                        st.image(api.image(e["url"]), width="stretch")
                    except api_client.ApiError:
                        st.caption("image missing")
                    credit = e.get("attribution") or e["source"]
                    st.caption(
                        f"{e['similarity']:.0%} similar · {credit} · {e.get('licence') or ''}"
                    )
            st.caption(f"Distance to training photos: {p['knn_distance']:.3f}")


def show_run(api: api_client.ApiClient, name: str, run: dict[str, Any]) -> None:
    upload = run["upload"]
    with st.container(border=True):
        st.markdown(f"**{name}**")
        if upload.get("note"):
            st.info(upload["note"])
        labels = view.detection_labels(run)
        frames = run["frames"]
        tips: list[str] = []
        left, right = st.columns([3, 2])
        with left:
            shown = frames if len(frames) <= 1 else frames[:: max(1, len(frames) // 4)]
            cols = st.columns(len(shown)) if len(shown) > 1 else [st.container()]
            for col, f in zip(cols, shown, strict=True):
                dets = [d for d in run["detections"] if d["frame_id"] == f["id"]]
                frame_bytes = api.image(f"/images/frames/{f['image_id']}")
                if upload["kind"] == "photo":
                    tips = view.photo_tips(frame_bytes, dets)
                with col:
                    st.image(view.draw_boxes(frame_bytes, dets, labels), width="stretch")
                    if upload["kind"] != "photo" or len(frames) > 1:
                        st.caption(f"{f['seconds']:.0f} s")
        with right:
            if not run["predictions"]:
                st.warning("No cat or dog found.")
            for p in run["predictions"]:
                show_prediction(api, run, p)
            for tip in tips:
                st.caption(f"📸 {tip}")
        faces = sum(f["faces_blurred"] for f in frames)
        if faces:
            st.caption(f"{faces} face(s) blurred in the stored copy.")


def upload_tab(api: api_client.ApiClient, limits: dict[str, float]) -> None:
    left, right = st.columns([3, 2], gap="large")
    with right:
        st.info(view.PHOTO_GUIDE, icon="📸")
        st.caption(view.limits_text(limits))
    with left:
        files = st.file_uploader(
            "Photos or short clips",
            type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "webm"],
            accept_multiple_files=True,
        )
        go = bool(files) and st.button("Identify", type="primary", icon="🔍")
    if files and go:
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


def _send(api: api_client.ApiClient, card: dict[str, Any], answer: str, label: str | None) -> None:
    shown = st.session_state.get("shown_at", {}).get(card["id"], time.time())
    try:
        api.review_answer(card["id"], answer, label, int((time.time() - shown) * 1000))
    except api_client.ApiError as exc:
        st.error(exc.detail)
        return
    st.session_state.pop("picking", None)
    st.rerun()


def review_tab(api: api_client.ApiClient, breeds: list[dict[str, str]]) -> None:
    top1, top2, top3 = st.columns([2, 3, 3], vertical_alignment="center")
    with top1:
        animals = {"Cats and dogs": None, "Cats": "cat", "Dogs": "dog"}
        animal = st.segmented_control(
            "Animals", list(animals), default="Cats and dogs", label_visibility="collapsed"
        )
    with top2:
        queues = {
            "All queues": None,
            "Web photos": "commons",
            "My batches": "batch",
            "Blind audit": "audit",
        }
        pick = st.selectbox("Queue", list(queues), label_visibility="collapsed")
    with top3:
        st.caption(view.stats_text(api.review_stats()))
    card = api.review_next(queues[pick], animals.get(animal or "Cats and dogs"))
    if card is None:
        st.success("Nothing left to review here. 🎉")
        return
    st.session_state.setdefault("shown_at", {}).setdefault(card["id"], time.time())
    with st.container(border=True):
        left, right = st.columns([3, 2], gap="large")
        with left:
            img = api.image(card["image_url"])
            if card.get("box"):
                x0, y0, x1, y1 = card["box"]
                box = {"id": card["id"], "species": "", "x0": x0, "y0": y0, "x1": x1, "y1": y1}
                img = view.draw_boxes(img, [box], {card["id"]: ""})
            st.image(img, width="stretch")
        with right:
            queue = view.QUEUE_NAMES.get(card["queue"], card["queue"])
            st.markdown(
                f"<div class='pl-muted'>{queue} · {card['species']} · {card['pending']} left</div>"
                f"<div class='pl-question'>{card['question']}</div>",
                unsafe_allow_html=True,
            )
            review_answers(api, card, breeds)
            refs = card.get("references", [])
            if refs:
                st.caption(f"{view.nice(card['suggested'])} examples from the training data")
                cols = st.columns(len(refs))
                for col, e in zip(cols, refs, strict=True):
                    with col:
                        try:
                            st.image(api.image(e["url"]), width="stretch")
                        except api_client.ApiError:
                            st.caption("image missing")
    st.caption(
        "Keys: → correct · ← wrong · ↑ not sure (click an empty spot first if the arrows "
        'switch tabs). "Not sure" deletes the photo: use it for poor photos.'
    )


def review_answers(
    api: api_client.ApiClient, card: dict[str, Any], breeds: list[dict[str, str]]
) -> None:
    blind = card["suggested"] is None
    picking = blind or st.session_state.get("picking") == card["id"]
    if not picking:
        b1, b2, b3 = st.columns(3)
        if b1.button("Wrong", shortcut="Left", width="stretch"):
            st.session_state["picking"] = card["id"]
            st.rerun()
        if b2.button("Not sure", shortcut="Up", width="stretch"):
            _send(api, card, "unsure", None)
        if b3.button("Correct", type="primary", shortcut="Right", width="stretch"):
            _send(api, card, "correct", None)
        return
    st.write("What is it?" if blind else "What is it instead?")
    for opt in card["options"]:
        if st.button(view.option_label(opt), key=f"opt-{opt}", width="stretch"):
            _send(api, card, "wrong", opt)
    names = {b["id"]: b["name"] for b in breeds if b["species"] == card["species"]}
    typed = st.selectbox(
        "Or type the breed",
        list(names),
        index=None,
        format_func=names.__getitem__,
        placeholder="Start typing a breed…",
        key=f"typed-{card['id']}",
    )
    if typed and st.button(f"Use {names[typed]}", type="primary", width="stretch"):
        _send(api, card, "wrong", typed)
    s1, s2 = st.columns(2)
    if s1.button("Not sure", shortcut="Up", width="stretch"):
        _send(api, card, "unsure", None)
    if s2.button("Skip", width="stretch"):
        _send(api, card, "skip", None)


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


def training_tab(api: api_client.ApiClient) -> None:
    st.caption(
        "Add a batch of photos of one breed for training. They are never shown as answers. "
        "Use your own photos, or ones you have permission or a Creative Commons licence for."
    )
    so_far = api.training_photo_counts()
    if so_far:
        st.caption(view.counts_text(so_far))
    breeds = api.breeds()
    names = {b["id"]: f"{b['name']} ({b['species']})" for b in breeds}
    batch = st.session_state.setdefault("train_batch", 0)
    breed = st.selectbox(
        "Breed of every photo in this batch",
        list(names),
        index=None,
        format_func=names.__getitem__,
        placeholder="Pick or type a breed",
        key=f"train-breed-{batch}",
    )
    files = st.file_uploader(
        "Photos",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key=f"train-files-{batch}",
        disabled=breed is None,
    )
    done = st.session_state.get("train_result")
    if done and done["batch"] == batch:
        show_training_result(done)
        if st.button("Start the next batch", type="primary"):
            st.session_state["train_batch"] = batch + 1
            st.session_state.pop("train_result", None)
            st.rerun()
        return
    if breed and files and st.button(f"Add {len(files)} photos as {names[breed]}", type="primary"):
        bar = st.progress(0.0, text="Checking photos…")
        results = []
        for i, f in enumerate(files, 1):
            try:
                r = api.add_training_photo(breed, f.name, f.getvalue())
            except api_client.ApiError as exc:
                r = {"status": "left_out", "reason": exc.detail}
            results.append((f.name, r))
            bar.progress(i / len(files), text=f"Checked {i} of {len(files)}")
        st.session_state["train_result"] = {
            "batch": batch,
            "breed": names[breed],
            "results": results,
        }
        st.rerun()


def show_training_result(done: dict[str, Any]) -> None:
    results = done["results"]
    by = {k: [n for n, r in results if r["status"] == k] for k in ("added", "already", "left_out")}
    st.success(
        f"{done['breed']}: added {len(by['added'])}, already there {len(by['already'])}, "
        f"left out {len(by['left_out'])}."
    )
    if by["left_out"]:
        with st.expander("Why some photos were left out"):
            for name, r in results:
                if r["status"] == "left_out":
                    st.write(f"**{name}**: {r['reason']}")


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("🐾 PetLens")
    st.caption("Cat and dog breeds, with a model that says when it isn't sure.")
    base_url = os.environ.get("PETLENS_API_URL", api_client.DEFAULT_URL)
    api = get_client(base_url)
    if not api.health():
        st.error(f"The API isn't answering at {base_url}. Start it with `make api`.")
        st.stop()
    limits = api.limits()
    upload, review, training, jobs = st.tabs(
        ["📷 Identify", "👆 Review", "🖼️ Training photos", "⚙️ Batch jobs"]
    )
    with upload:
        upload_tab(api, limits)
    with review:
        review_tab(api, api.breeds())
    with training:
        training_tab(api)
    with jobs:
        jobs_tab(api)


main()
