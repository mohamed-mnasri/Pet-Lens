# 0006. Group by appearance, not with a tracker model

- **Status:** Accepted
- **Date:** 2026-11-22

## Context

A clip or a webcam feed shows the same animal in many frames, and the answer should be one per animal, not one per frame. Object trackers (ByteTrack, DeepSORT) follow boxes by motion and need tuning for frame rate and occlusion. PetLens samples a few frames from a clip and gets the webcam at a variable rate, so motion between frames is not reliable.

## Decision

Follow animals by **appearance**: a detection joins the group of the same species whose recent DINOv2 embedding is most similar, above a threshold, otherwise it starts a new group. Clips group all sampled frames at once; the live page does it frame by frame, with a running average embedding and the decision made on probabilities averaged over the last 5 frames.

## Consequences

- No extra model or dependency; the embedding is already computed for the classifier.
- Works at any frame rate and across cuts, and the same idea matches animals to known pets across uploads.
- Two near-identical animals in one frame can be merged into one group. That is rare in my footage and visible on the page.
