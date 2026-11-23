# 0008. The live camera page is plain HTML and JavaScript

- **Status:** Accepted
- **Date:** 2026-11-23

## Context

Streamlit reruns the whole script on each interaction and isn't made for a 20 frames-per-second loop. Components such as streamlit-webrtc add WebRTC, a TURN-style setup and more packages, for a page that only ever runs on localhost.

## Decision

The API serves one static page (`/live`). The browser takes the webcam with `getUserMedia`, sends a 640 px JPEG to `POST /live/frame`, draws the boxes and answers on a canvas, and sends the next frame when the answer arrives. The server keeps a small session per tab (tracks, smoothing, storage clock).

## Consequences

- No new dependency; the page is about 170 lines.
- The FPS shown is the real end-to-end rate, including the round trip, because frames are sent one at a time.
- The API is the only place that touches the GPU, so it can refuse frames while a VLM job runs.
- The page looks different from the Streamlit app; it is linked from it with one button.
