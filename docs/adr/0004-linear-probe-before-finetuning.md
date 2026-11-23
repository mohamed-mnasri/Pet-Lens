# 0004. A DINOv2 linear probe first, partial fine-tuning only if it earns its place

- **Status:** Accepted
- **Date:** 2026-11-22

## Context

The GPU is an RTX 2060 with 6 GB. The training set is a few thousand photos per species, and the labels from Commons are noisy. Full fine-tuning of a ViT on that is slow, easy to overfit, and makes every experiment expensive.

## Decision

- Start with **frozen DINOv2 ViT-S/14 embeddings and per-species linear or small MLP heads**, temperature-scaled on a calibration split. Embeddings are cached, so a head trains in seconds.
- Use the same embeddings for the kNN open-set score and the evidence panel.
- Try **partial fine-tuning** (the last 2 blocks and the final norm, fp16, webcam-style degradation) and ViT-B/14 as ablations, and keep one only if it passes the promotion gate against frozen ViT-S (ADR 0007).

## Consequences

- Fast iteration: new data or a new policy is minutes, not hours.
- The kNN distance and the evidence photos stay meaningful because the space is a general-purpose one.
- A fine-tuned backbone changes the embedding space, so a new run needs its own embedding cache and evidence index; runs record which backbone they used.
- The ablation report picks the winner by golden-set gain, not by in-domain accuracy, which a fine-tune can inflate.
