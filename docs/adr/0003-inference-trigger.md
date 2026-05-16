# ADR-0003: Inference trigger strategy

- **Status:** Provisional — final decision deferred to Phase 1.5
- **Date:** 2026-05-16
- **Deciders:** Naoya Higashitani

## Context

`POST /infer` is pull-based: a caller has to ask for inference on a specific
machine + sensor. Phase 1.3 deliberately stops there. Phase 1.4 (Streamlit
dashboard) does not change this — the dashboard only **displays** results
that already exist.

But "real-time anomaly detection" only feels real when scoring runs
*automatically* as data arrives. Some component must decide *when* to call
`/infer`. The question is which.

## Options under consideration

| Option | Where the trigger lives | Pros | Cons |
|--------|------------------------|------|------|
| **A. Auto-trigger inside `/ingest`** | `FastAPI BackgroundTasks` after each successful insert | Zero new infrastructure; "freshest" inference | Couples ingest latency to inference; on burst writes, fires N inference jobs for what could be one batched call; harder to back-pressure |
| **B. Dedicated scheduler container** | Cron-like service that POSTs to `/infer` every N seconds | Cleanly decoupled; trivially horizontal-scalable; can batch | Extra container; adds latency floor; configuration sprawl |
| **C. Message queue + worker** | `/ingest` enqueues; a worker dequeues, scores, writes back | Most cloud-native; introduces Phase 3 muscle (SQS / Azure Service Bus) early | Heaviest infra; overkill for a single-machine MVP |
| **D. Dashboard-driven (current state)** | Streamlit's "Generate demo data" button calls `/ingest` then `/infer` | No new infra; visible to the user; works today | Not "real-time" outside the demo button; useless for unattended operation |

## Decision

For **Phase 1.4**: keep option **D**. The Streamlit dashboard exposes a
"Generate demo data" button that calls `/ingest` followed by `/infer` so
the rest of the UI has something to render. This is acceptable because
Phase 1.4's goal is "the UI renders correctly", not "the system runs unattended".

For **Phase 1.5** (local integration via docker-compose): the question becomes
real. The decision will be made there with a follow-up ADR (likely **B** —
a small `ingester` + `scorer` service pair — to keep API behaviour pure and
demonstrate clean separation of concerns). Option **A** would also be
acceptable; rejected primarily to keep API endpoints free of side effects.

## Consequences

- Phase 1.4 ships without any automated inference. The README must
  explicitly say "click the demo button to populate data".
- Phase 1.5 ADR will need to specify: trigger cadence, batching policy,
  back-pressure behaviour, and what happens when the model is unavailable
  (today: 503; with auto-trigger: silent skip + logged warning).
- This ADR is **provisional** and will be superseded by a follow-up; it
  exists so that future readers can trace why Phase 1.4 looks the way it does.
