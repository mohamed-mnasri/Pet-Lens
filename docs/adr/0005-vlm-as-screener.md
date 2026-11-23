# 0005. The VLM screens photos; it never sets a label

- **Status:** Accepted
- **Date:** 2026-11-23

## Context

A local vision-language model (`qwen3-vl:4b` through Ollama) can describe photos. It is tempting to let it label breeds or audit the dataset's labels. But a 4B model's breed knowledge is unmeasured, its mistakes would flow into training without anyone noticing, and it needs most of the GPU.

## Decision

- The VLM answers **narrow yes or no questions** about each Commons and Openverse photo, in JSON validated by Pydantic: real photo or drawing or toy, whole animal visible, kitten, blurry, several animals. One retry, then the answer is stored as `parse_failed` and counted.
- Its tags only **keep weak photos out of training** (not a photo, partial, several animals), unless I reviewed the photo myself. They never change a label.
- Its tags are **measured** against my reviews and the image-quality flags (Cohen's kappa), and become failure slices in the marts.
- Naming breeds directly is run as a separate, reported **experiment**, not used in the product.
- It runs only as a worker job, never next to the live camera.

## Consequences

- A bad VLM answer can at worst drop a training photo, never teach a wrong label.
- Tags are cached per prompt version and model digest, so a new prompt is a new, comparable run.
- Screening thousands of photos takes hours on the 2060; it runs overnight and resumes where it stopped.
