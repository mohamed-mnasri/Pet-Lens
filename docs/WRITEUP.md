# PetLens: engineering write-up

What I built, how the pieces fit, what worked and what didn't. The design is in [SPEC.md](SPEC.md), the order of work and implementation notes in [PLAN.md](PLAN.md), and the numbers for the model the app serves in [MODEL_CARD.md](../MODEL_CARD.md).

## The problem

Breed classifiers usually answer every photo with a breed. That is wrong for most real cats, which have no breed at all, and for anything outside the classes the model knows. I wanted a recogniser that is useful on my own pets and footage, and honest about uncertainty: it should say "domestic cat, no clear breed" or "not sure" when that is the right answer, and its confidence should mean something.

The constraints shaped everything: one Windows laptop with an RTX 2060 (6 GB), no cloud services, €0, and only openly licensed data.

## How it works

1. **Ingest.** Uploads are checked for size, type and pixel count before anything is decoded, then stored as 1024 px WebP. Rejected inputs go to a dead-letter table.
2. **Detect.** RF-DETR finds cats, dogs and people. People are blurred before anything is stored.
3. **Embed and classify.** Each animal crop goes through DINOv2 (ViT-S/14 frozen by default). A per-species head gives breed probabilities, temperature-scaled on a calibration split.
4. **Decide.** A kNN distance to the training embeddings measures how unusual the crop is. A decision policy, tuned on held-out data against a false breed claim limit, chooses *accept*, *no clear breed* or *review*. Look-alike pairs are reported together when the gap is small.
5. **Explain.** The evidence panel shows the nearest training photos from pgvector.
6. **Group.** Frames of a clip, and webcam frames over time, are grouped by appearance; uploads are matched to known pets the same way.
7. **Learn.** I swipe answers right or wrong. Labels are append-only events with their origin. A new dataset version trains a new run, which is promoted only through the gate.

## Decisions that mattered

The ADRs in [adr/](adr) give the full reasoning; the short version:

- **Frozen embeddings first** ([ADR 0004](adr/0004-linear-probe-before-finetuning.md)). Training a head takes seconds, so I could iterate on data and policy instead of waiting on GPU runs. Fine-tuning is an ablation that has to beat the frozen baseline through the gate.
- **The decision policy is the product.** Accuracy on accepted answers and the false breed claim rate are tuned explicitly, and the policy is stored with each run, so a run always serves with the thresholds it was tuned with.
- **One database** ([ADR 0002](adr/0002-postgres-over-duckdb.md), [ADR 0003](adr/0003-job-queue-in-postgres.md)). Postgres holds the app data, the event logs, the job queue, the vectors and the dbt marts. Triggers keep the event tables append-only.
- **Promotion by a paired test** ([ADR 0007](adr/0007-paired-promotion-gate.md)). A frozen golden set plus sources never trained on, a paired bootstrap over images, a per-breed recall floor and no rise in false breed claims.
- **The VLM as a screener, not a labeller** ([ADR 0005](adr/0005-vlm-as-screener.md)). Its worst mistake can drop a training photo, never teach a wrong label, and its agreement with my reviews is measured.
- **Appearance grouping instead of tracking** ([ADR 0006](adr/0006-appearance-grouping-not-tracking.md)) and **a plain HTML live page** ([ADR 0008](adr/0008-live-page-plain-html.md)) kept the dependency list short and the camera loop honest about its frame rate.

## Data engineering

- **Manifests, not folders.** Each dataset version is a committed Parquet manifest: source, licence, author, label origin, quality flags, group and split for every photo. Images are downloaded, never committed, and any version can be rebuilt.
- **Grouped splits.** Near-duplicates (perceptual hash) and photos of the same animal stay in one split, and a leak check against the test sets runs on every build.
- **Held-out sets with a purpose.** Domestic cats count false breed claims, unknown breeds check rejection, unseen sources check generalisation, and the golden set is frozen for promotion.
- **dbt marts with contracts.** Evaluation results, predictions, reviews, the registry, VLM tags and pet identities are modelled in staging, intermediate and mart layers, with enforced contracts, snapshots for label and pet history, and unit tests. The Model health tab reads only marts.
- **CI.** Ruff, strict mypy, pytest with a Postgres+pgvector service, and a `dbt build` on fixtures, on every push.

## What worked

- Treating "no clear breed" as a trained class with its own data, instead of a low-confidence fallback, did more for real-world answers than any backbone change.
- Cached embeddings made the whole loop (new labels, new policy, new evaluation) fast enough to run after every review session.
- Storing every label and pet change as an event made it easy to answer "why does the model think this?" and to replay matching with new thresholds.
- Having the gate before the fine-tuning work kept me from chasing in-domain gains that didn't carry over to new sources.

## What didn't, or not yet

- **Commons categories are noisy and uneven.** Several breed categories were nearly empty, and some photos are drawings, toys or several animals. Screening (quality flags, the VLM, my reviews) is what made them usable, and thin breeds still have wide intervals.
- **Look-alike breeds** remain the main source of errors. Showing both candidates is more honest than forcing one.
- **The VLM is slow on 6 GB.** Screening runs overnight and can't run while the camera is open.
- **Pet matching by appearance** can merge two very similar animals, so merge and split are one click and every change is logged.

## Reproducing it

`make up && make demo` shows the app on a seeded database with no dataset or GPU. The full pipeline (data, embeddings, training, gate, evaluation, marts, cards) is the target list in the [README](../README.md#the-full-pipeline).
