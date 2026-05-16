# ADR-0004: Inference trigger — final decision

- **Status:** Accepted (supersedes the provisional half of [ADR-0003](0003-inference-trigger.md))
- **Date:** 2026-05-16
- **Deciders:** Naoya Higashitani

## Context

ADR-0003 left the choice of inference-trigger architecture open for Phase 1.5.
Phase 1.5 brings up the full docker-compose stack, at which point the question
becomes urgent: *what process actually calls `/infer` once data starts
flowing?*

ADR-0003 leaned toward Option B (dedicated scheduler container); this ADR
locks that in and records the operational details so the next reader does
not have to re-derive them.

## Decision

Run two new sidecar containers next to `api` and `dashboard`:

### `ingester`
- Generates a fresh batch of synthetic readings every `INGESTER_INTERVAL_SECONDS`
  (default `10`).
- Each batch is `INGESTER_BATCH_SIZE` samples (default `30`) with random
  spike injection so the dashboard has *something interesting* to render.
- Hits `POST /ingest` directly. Sleeps. Repeats.
- On any API error: log a warning and continue.

### `scorer`
- Hits `POST /infer` every `SCORER_INTERVAL_SECONDS` (default `5`).
- Specifies `machine_id` + `sensor_name` from env (defaults match the
  ingester's writes so the two are paired by configuration).
- HTTP 422 (not enough data yet) and 503 (model still loading) are
  expected during startup and are logged at WARN, not ERROR. The process
  keeps running.

## Alternatives revisited

| Option | Why rejected (or deferred) |
|--------|----------------------------|
| **A. Auto-trigger on `/ingest`** via FastAPI `BackgroundTasks` | Couples ingest latency to inference. CRUD endpoints with side effects are harder to test, reason about, and rate-limit. Saved as a possible future toggle for "low-volume edge" deployments where every reading should be scored immediately. |
| **C. Message queue + worker** (SQS / Azure Service Bus) | Right answer for production at scale. Overkill for an MVP and would crowd out the Phase 3 microservice story, where MQ-based async actually earns its keep. |
| **D. Dashboard-driven only** (the Phase 1.4 stop-gap) | Was useful for demos but fails the "real-time platform" framing the moment the dashboard is closed. |

## Operational details

- **Crash policy:** `restart: unless-stopped` for both sidecars. They are
  designed to be idempotent — restarting mid-batch loses at most one tick.
- **Observability:** Both processes use `structlog` JSON output through
  stdout, so `docker compose logs scorer` is structured and grep-friendly.
- **Configuration:** Every interval and batch size is an env var. No
  recompile / rebuild needed to retune.
- **Multi-machine support (future):** today both sidecars are pinned to a
  single `MACHINE_ID` / `SENSOR_NAME`. Scaling to N machines is a loop
  inside the container, not a topology change.

## Consequences

- API endpoints remain side-effect-free. Tests stay clean.
- Two new containers in the compose stack — `docker compose ps` shows what
  is running and `docker compose logs scorer` shows the trigger cadence.
- The "real-time" feel works without the dashboard open. Anyone hitting
  the API directly sees a continuously growing `anomaly_results` table.
- If Phase 3 introduces a message queue, ingester / scorer will be the
  first two services to migrate; the trigger logic moves to the queue
  consumer but the *shape* (small dedicated process, idempotent, restart-safe)
  stays the same.
