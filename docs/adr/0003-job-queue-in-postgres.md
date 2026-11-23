# 0003. The job queue lives in Postgres

- **Status:** Accepted
- **Date:** 2026-11-22

## Context

Some work is too slow for a request: reviewing a folder of photos, screening thousands of dataset photos with the VLM. It needs to survive restarts, report progress to the page, and retry or park failures. The usual answer is Redis with Celery or RQ.

## Decision

A `job` table in Postgres, claimed with `SELECT ... FOR UPDATE SKIP LOCKED`, run by one worker process (`make worker`). Failed jobs retry with a growing delay up to `max_attempts` and are then marked dead with the error kept. Rejected inputs go to a `dead_letter` table. Progress is a pair of columns the page polls.

## Consequences

- No extra service on a single laptop; the queue is in the same backups and transactions as the data it changes.
- `SKIP LOCKED` makes a second worker safe if I ever start one.
- Throughput is far below a real broker, which doesn't matter at a few jobs an hour.
- The camera checks this table to refuse frames while a VLM job holds the GPU (see ADR 0008).
