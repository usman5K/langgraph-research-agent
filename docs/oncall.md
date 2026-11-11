# Kestrel Analytics: On-call Handbook

## Rotation
On-call runs in weekly rotations starting Monday 10:00 PKT. Each rotation has a primary and a
secondary engineer. The primary must acknowledge a page within 5 minutes; if not, it escalates
to the secondary after 10 minutes and to the engineering manager after 20 minutes.

## Severity levels
- SEV1: customer-facing outage or data loss. Open an incident channel and page the incident commander.
- SEV2: degraded performance for more than 10% of requests. Fix within 4 hours.
- SEV3: internal tooling issues. Fix within 2 business days.

## After an incident
Every SEV1 and SEV2 gets a blameless postmortem within 3 business days, with action items tracked
in the "reliability" board.
