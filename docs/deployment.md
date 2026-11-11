# Kestrel Analytics: Deployment Guide

## Pipeline
Every merge to `main` builds a Docker image, runs the test suite and deploys to staging.
Production deploys are promoted from staging after a 30-minute soak with no new alerts.

## Deploy windows
Production deploys happen Monday to Thursday, 11:00 to 17:00 PKT. No deploys on Fridays or on
the day before a public holiday unless it is a SEV1 fix.

## Rollbacks
Rollbacks use the previous image tag and must complete within 10 minutes. Database migrations must
be backwards compatible for one release so that a rollback never requires a down-migration.
