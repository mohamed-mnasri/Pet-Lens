"""Group detections across the frames of a clip by appearance. No tracker.

Two detections join the same group when they are the same species, come from
different frames, their DINOv2 embeddings are similar enough, and their box sizes are
plausible for the same animal. Detections in the same frame are never merged, so two
cats side by side stay two groups. If a detection fits two groups almost equally well,
the grouping is ambiguous and the caller falls back to per-frame results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GroupingConfig:
    min_similarity: float = 0.65
    max_size_ratio: float = 3.0
    ambiguity_margin: float = 0.05


@dataclass(frozen=True)
class Item:
    frame: int
    species: str
    embedding: NDArray[np.float32]
    area: float


@dataclass
class Group:
    species: str
    members: list[int] = field(default_factory=list)  # indices into the item list
    frames: set[int] = field(default_factory=set)
    _sum: NDArray[np.float64] | None = None
    areas: list[float] = field(default_factory=list)

    def centroid(self) -> NDArray[np.float64]:
        assert self._sum is not None
        out: NDArray[np.float64] = self._sum / max(np.linalg.norm(self._sum), 1e-12)
        return out

    def add(self, idx: int, item: Item) -> None:
        v = item.embedding.astype(np.float64)
        v = v / max(float(np.linalg.norm(v)), 1e-12)
        self._sum = v if self._sum is None else self._sum + v
        self.members.append(idx)
        self.frames.add(item.frame)
        self.areas.append(item.area)


@dataclass(frozen=True)
class GroupingResult:
    groups: list[Group]
    ambiguous: bool
    reason: str | None = None


def group_items(items: list[Item], cfg: GroupingConfig = GroupingConfig()) -> GroupingResult:
    groups: list[Group] = []
    ambiguous_reasons: list[str] = []
    order = sorted(range(len(items)), key=lambda i: items[i].frame)
    for idx in order:
        item = items[idx]
        v = item.embedding.astype(np.float64)
        v = v / max(float(np.linalg.norm(v)), 1e-12)
        scored: list[tuple[float, int]] = []
        for g_idx, g in enumerate(groups):
            if g.species != item.species or item.frame in g.frames:
                continue
            median_area = float(np.median(g.areas))
            ratio = max(item.area, 1e-9) / max(median_area, 1e-9)
            if not (1 / cfg.max_size_ratio <= ratio <= cfg.max_size_ratio):
                continue
            sim = float(v @ g.centroid())
            if sim >= cfg.min_similarity:
                scored.append((sim, g_idx))
        scored.sort(reverse=True)
        if len(scored) >= 2 and scored[0][0] - scored[1][0] < cfg.ambiguity_margin:
            ambiguous_reasons.append(
                f"frame {item.frame}: a {item.species} looks equally like two animals "
                f"(similarity {scored[0][0]:.2f} vs {scored[1][0]:.2f})"
            )
        if scored:
            groups[scored[0][1]].add(idx, item)
        else:
            g = Group(item.species)
            g.add(idx, item)
            groups.append(g)
    if ambiguous_reasons:
        return GroupingResult(groups, True, "; ".join(ambiguous_reasons))
    return GroupingResult(groups, False)
