"""DINOv2 embeddings for every image crop, cached as Parquet.

Torch and timm are optional (``uv sync --extra ml``) and imported only when the model
is actually loaded, so CI and the data pipeline don't need them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from PIL import Image

from petlens.data.imaging import stored_path
from petlens.data.manifest import ManifestRow

log = logging.getLogger(__name__)

DEFAULT_MODEL = "vit_small_patch14_dinov2.lvd142m"
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

EmbedFn = Callable[[list[Image.Image]], NDArray[np.float32]]


def crop_for_model(
    img: Image.Image, bbox: Sequence[float] | None, margin: float = 0.10, size: int = IMAGE_SIZE
) -> Image.Image:
    """Crop to the normalised box plus a margin, pad to a square and resize.

    Without a box the whole image is used. Padding uses the image's mean colour so the
    model doesn't see hard black bars.
    """
    img = img.convert("RGB")
    w, h = img.size
    if bbox is not None:
        x0, y0, x1, y1 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
        mx, my = (x1 - x0) * margin, (y1 - y0) * margin
        box = (
            max(0, int(x0 - mx)),
            max(0, int(y0 - my)),
            min(w, round(x1 + mx)),
            min(h, round(y1 + my)),
        )
        if box[2] > box[0] and box[3] > box[1]:
            img = img.crop(box)
    cw, ch = img.size
    side = max(cw, ch)
    fill = tuple(int(v) for v in np.asarray(img).reshape(-1, 3).mean(axis=0))
    square = Image.new("RGB", (side, side), fill)
    square.paste(img, ((side - cw) // 2, (side - ch) // 2))
    return square.resize((size, size), Image.Resampling.BICUBIC)


FINETUNED_PREFIX = "ft-"


def finetuned_dir(model_name: str, models_dir: Path | None = None) -> Path | None:
    """Folder of a fine-tuned backbone (Step 12), or None for a stock timm model."""
    if not model_name.startswith(FINETUNED_PREFIX):
        return None
    if models_dir is None:
        from petlens.config import get_settings

        models_dir = get_settings().models_dir
    return models_dir / "finetune" / model_name


def create_backbone(model_name: str, models_dir: Path | None = None) -> Any:
    """A timm DINOv2 backbone with no classifier: stock weights, or a fine-tuned copy."""
    import json

    import timm
    import torch

    ft = finetuned_dir(model_name, models_dir)
    if ft is None:
        return timm.create_model(model_name, pretrained=True, num_classes=0, img_size=IMAGE_SIZE)
    base = json.loads((ft / "config.json").read_text(encoding="utf-8"))["base"]
    model: Any = timm.create_model(base, pretrained=False, num_classes=0, img_size=IMAGE_SIZE)
    model.load_state_dict(torch.load(ft / "backbone.pt", map_location="cpu", weights_only=True))
    return model


def load_dinov2(
    model_name: str = DEFAULT_MODEL, device: str | None = None, models_dir: Path | None = None
) -> EmbedFn:
    """Return a function mapping a batch of 224 px RGB images to L2-normalised embeddings."""
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model: Any = create_backbone(model_name, models_dir)
    model.eval().to(dev)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    log.info("loaded %s on %s", model_name, dev)

    def embed(batch: list[Image.Image]) -> NDArray[np.float32]:
        arr = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in batch])
        with torch.inference_mode():
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(dev)
            x = (x - mean) / std
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                feats = model(x)
            feats = torch.nn.functional.normalize(feats.float(), dim=1)
            out: NDArray[np.float32] = np.asarray(feats.cpu().numpy(), dtype=np.float32)
        return out

    return embed


def cache_path(data_dir: Path, model_name: str) -> Path:
    return data_dir / "processed" / "embeddings" / model_name / "cache.parquet"


def read_embeddings(path: Path) -> dict[str, NDArray[np.float32]]:
    if not path.exists():
        return {}
    table = pq.read_table(path)
    ids = table.column("image_id").to_pylist()
    vecs = table.column("embedding").to_pylist()
    return {i: np.asarray(v, dtype=np.float32) for i, v in zip(ids, vecs, strict=True)}


def write_embeddings(emb: dict[str, NDArray[np.float32]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted(emb)
    dim = len(emb[ids[0]]) if ids else 0
    table = pa.table(
        {
            "image_id": pa.array(ids, pa.string()),
            "embedding": pa.array([emb[i].tolist() for i in ids], pa.list_(pa.float32(), dim)),
        }
    )
    tmp = path.with_suffix(".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


def embed_rows(
    rows: list[ManifestRow],
    data_dir: Path,
    embed_fn: EmbedFn,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
) -> dict[str, NDArray[np.float32]]:
    """Embed every row not already in the cache. Returns embeddings for all ``rows``."""
    path = cache_path(data_dir, model_name)
    cache = read_embeddings(path)
    todo = [r for r in rows if r.image_id not in cache]
    images = data_dir / "processed" / "images"
    log.info("embeddings: %d cached, %d to compute", len(rows) - len(todo), len(todo))
    for start in range(0, len(todo), batch_size):
        chunk = todo[start : start + batch_size]
        crops = []
        for r in chunk:
            with Image.open(stored_path(images, r.image_id)) as im:
                crops.append(crop_for_model(im, r.bbox))
        for r, v in zip(chunk, embed_fn(crops), strict=True):
            cache[r.image_id] = v
        if (start // batch_size) % 20 == 19:
            write_embeddings(cache, path)  # checkpoint, so an interrupted run resumes
            log.info("embedded %d / %d", start + len(chunk), len(todo))
    if todo:
        write_embeddings(cache, path)
    return {r.image_id: cache[r.image_id] for r in rows}
