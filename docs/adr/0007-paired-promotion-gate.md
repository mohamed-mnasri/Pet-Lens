# 0007. A paired bootstrap gate decides promotion

- **Status:** Accepted
- **Date:** 2026-11-23

## Context

Each new dataset version or backbone gives a new run, and it is easy to fool myself with a one-point gain on a test set of a few hundred photos. Comparing two accuracy numbers with their own intervals is also too weak: the two runs answer the same photos, so their errors are correlated.

## Decision

A challenger replaces the champion only if all of these hold, on photos neither run trained on:

- **Higher top-1 on the frozen golden set and on sources never trained on**, with the 5th percentile of a paired bootstrap over images (10,000 resamples) above zero.
- **No breed with 10 or more photos loses more than 10 points of recall.**
- **The false breed claim rate on domestic cats doesn't rise**, each run with its own tuned policy.
- **No golden photo is in the training pool.**

The golden set is frozen once (`make golden`) and refuses to be rewritten. Every gate result is stored in the registry and written to `gate.md`.

## Consequences

- Promotion is reproducible and explainable: the numbers that justified it are in the registry and on the Model health tab.
- Small real gains can fail the gate on a small golden set; that is the intended bias.
- The app always serves the champion, so an experiment can never reach the page by accident.
