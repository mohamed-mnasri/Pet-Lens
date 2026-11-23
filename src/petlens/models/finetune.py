"""Fine-tune DINOv2 with its last blocks unfrozen (Step 12 ablations).

One backbone is tuned for both species with a joint linear layer over every cat and dog
class, so the app keeps a single embedder. Training uses the same photos, splits and
seed as the frozen baseline (the same training pool, golden photos left out), fp16 on
the GPU, and augmentation that never touches colour hue (coat colour is a breed cue):
random resized crop, flip, small rotation, mild brightness and contrast, webcam-style
downscaling and JPEG, and random erasing. The epoch with the best validation balanced
accuracy is kept.

The tuned backbone then becomes an embedding model of its own (``ft-...``): every photo
is embedded with it and the usual heads, calibration and decision policy are trained
on top (``petlens.models.train``), so the run goes through the same evaluation,
registry and promotion gate as any other. Torch and timm come from the ml extra.
"""

from __future__ import annotations

import io
import json
import logging
import random
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from petlens.data.catalog import BreedCatalog
from petlens.data.imaging import stored_path
from petlens.data.manifest import ManifestRow
from petlens.models import embed
from petlens.models.train import in_training_pool

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinetuneConfig:
    base: str = embed.DEFAULT_MODEL
    unfreeze_blocks: int = 2
    epochs: int = 6
    batch_size: int = 32
    lr_backbone: float = 2e-5
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    warmup_share: float = 0.1
    patience: int = 2
    seed: int = 0
    num_workers: int = 4


SHORT = {
    "vit_small_patch14_dinov2.lvd142m": "vits14",
    "vit_base_patch14_dinov2.lvd142m": "vitb14",
}


def model_name(cfg: FinetuneConfig, dataset_version: str) -> str:
    short = SHORT.get(cfg.base, cfg.base.split(".")[0])
    return f"{embed.FINETUNED_PREFIX}{short}-last{cfg.unfreeze_blocks}-{dataset_version}"


class WebcamDegrade:
    """Sometimes shrink the photo and save it as a low-quality JPEG, like a webcam frame."""

    def __init__(self, p: float = 0.3) -> None:
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        w, h = img.size
        side = random.randint(96, 160)
        small = img.resize((side, max(1, round(side * h / w))), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=random.randint(40, 80))
        buf.seek(0)
        with Image.open(buf) as degraded:
            return degraded.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)


def train_transform() -> Any:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(embed.IMAGE_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            WebcamDegrade(),
            transforms.ToTensor(),
            transforms.Normalize(embed.MEAN, embed.STD),
            transforms.RandomErasing(p=0.25),
        ]
    )


def eval_transform() -> Any:
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(embed.MEAN, embed.STD)])


class CropDataset(Dataset[tuple[Any, int]]):  # type: ignore[misc,unused-ignore]
    """Animal crops from the stored dataset images, as in ``make embed``."""

    def __init__(
        self,
        items: Sequence[tuple[str, Sequence[float] | None, int]],
        images_dir: Path,
        transform: Any,
        size: int,
    ) -> None:
        self.items = list(items)
        self.images_dir = images_dir
        self.transform = transform
        self.size = size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[Any, int]:
        image_id, bbox, label = self.items[i]
        with Image.open(stored_path(self.images_dir, image_id)) as im:
            crop = embed.crop_for_model(im, bbox, size=self.size)
        return self.transform(crop), label


class Tuner(nn.Module):  # type: ignore[misc,unused-ignore]
    """The backbone with only its last ``unfreeze`` blocks (and final norm) trainable,
    plus a linear layer over every class. Frozen blocks keep no activations for backprop."""

    def __init__(self, backbone: Any, unfreeze: int, n_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        for p in backbone.parameters():
            p.requires_grad = False
        if unfreeze:
            for block in backbone.blocks[-unfreeze:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in backbone.norm.parameters():
                p.requires_grad = True
        self.head = nn.Linear(backbone.num_features, n_classes)

    def forward(self, x: Any) -> Any:
        return self.head(self.backbone(x))


def joint_classes(breeds: BreedCatalog) -> list[str]:
    return [f"{s}:{c}" for s in ("cat", "dog") for c in breeds.training_classes(s)]


def split_items(
    rows: list[ManifestRow],
    classes: list[str],
    admitted: frozenset[str],
    golden: frozenset[str],
    split: str,
) -> list[tuple[str, Sequence[float] | None, int]]:
    index = {c: i for i, c in enumerate(classes)}
    out: list[tuple[str, Sequence[float] | None, int]] = []
    for r in rows:
        key = f"{r.species}:{r.breed}"
        if r.split != split or key not in index or r.image_id in golden:
            continue
        if split in ("train", "val") and not in_training_pool(r, admitted):
            continue
        out.append((r.image_id, r.bbox, index[key]))
    return out


def balanced_accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    recalls = [float((pred[y == c] == c).mean()) for c in np.unique(y)]
    return float(np.mean(recalls)) if recalls else 0.0


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def finetune(
    rows: list[ManifestRow],
    breeds: BreedCatalog,
    data_dir: Path,
    out_dir: Path,
    cfg: FinetuneConfig,
    admitted: frozenset[str] = frozenset(),
    golden: frozenset[str] = frozenset(),
    device: str | None = None,
    backbone: Any = None,
) -> dict[str, Any]:
    """Tune and save ``out_dir/backbone.pt`` and ``config.json``. Returns the history.
    Tests pass a tiny ``backbone``; otherwise the stock DINOv2 weights are loaded."""
    _seed_everything(cfg.seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    classes = joint_classes(breeds)
    images = data_dir / "processed" / "images"
    train_items = split_items(rows, classes, admitted, golden, "train")
    val_items = split_items(rows, classes, admitted, golden, "val")
    if not train_items or not val_items:
        raise SystemExit("no training or validation photos; run `make data-build` first")
    model = Tuner(backbone or embed.create_backbone(cfg.base), cfg.unfreeze_blocks, len(classes))
    model.to(dev)
    counts = np.bincount([t[2] for t in train_items], minlength=len(classes)).astype(np.float64)
    weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / len(classes), 0.0)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=dev),
        label_smoothing=cfg.label_smoothing,
    )
    groups = [{"params": model.head.parameters(), "lr": cfg.lr_head}]
    tuned = [p for p in model.backbone.parameters() if p.requires_grad]
    if tuned:
        groups.append({"params": tuned, "lr": cfg.lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    loader_kw: dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": dev == "cuda",
        "persistent_workers": cfg.num_workers > 0,
    }
    train_dl = DataLoader(
        CropDataset(train_items, images, train_transform(), 256),
        shuffle=True,
        drop_last=len(train_items) > cfg.batch_size,
        generator=torch.Generator().manual_seed(cfg.seed),
        **loader_kw,
    )
    val_dl = DataLoader(
        CropDataset(val_items, images, eval_transform(), embed.IMAGE_SIZE), **loader_kw
    )
    steps = max(1, cfg.epochs * len(train_dl))
    warmup = max(1, round(steps * cfg.warmup_share))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))),
    )
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    history: list[dict[str, float]] = []
    best: tuple[float, dict[str, Any]] | None = None
    stale = 0
    started = time.monotonic()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss.detach()) * len(y)
            seen += len(y)
        model.eval()
        preds, labels = [], []
        with torch.inference_mode():
            for x, y in val_dl:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model(x.to(dev))
                preds.append(logits.argmax(1).cpu().numpy())
                labels.append(y.numpy())
        val_acc = balanced_accuracy(np.concatenate(preds), np.concatenate(labels))
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1), "val_bal_acc": val_acc})
        log.info("epoch %d: loss %.4f, val balanced accuracy %.4f", epoch, total / seen, val_acc)
        if best is None or val_acc > best[0]:
            state = {k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()}
            best, stale = (val_acc, state), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                log.info("no gain for %d epochs: stopping", stale)
                break
    assert best is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best[1], out_dir / "backbone.pt")
    info = {
        "base": cfg.base,
        "config": asdict(cfg),
        "classes": len(classes),
        "train_images": len(train_items),
        "val_images": len(val_items),
        "best_val_bal_acc": best[0],
        "history": history,
        "seconds": round(time.monotonic() - started, 1),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if use_amp else None,
        "device": dev,
    }
    (out_dir / "config.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def reset_cache(data_dir: Path, name: str) -> None:
    """A refit of the same name must not reuse embeddings from the previous weights."""
    folder = embed.cache_path(data_dir, name).parent
    if folder.exists():
        shutil.rmtree(folder)
