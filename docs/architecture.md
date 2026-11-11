# Kestrel Analytics: Architecture Overview

## Services
The platform has three Python services: `ingest` (FastAPI, receives events), `pipeline`
(Celery workers that enrich and aggregate events) and `insights` (FastAPI, serves dashboards and
an LLM-powered "ask your data" feature).

## Data stores
PostgreSQL 16 is the system of record. Embeddings for the "ask your data" feature live in the same
cluster via the PGVector extension. Redis is used for Celery queues and caching.

## SLOs
The `ingest` API targets 99.9% monthly availability and p95 latency under 200 ms. The `insights`
API targets 99.5% availability and p95 latency under 800 ms.
