<!--
Keep this short. The interesting record is the commit message + the linked phase doc / ADR.
-->

## Summary

<!-- One or two sentences. What does this change do, and why now? -->

## Phase / scope

<!-- e.g. "Phase 1.8 — multivariate features" or "infra: pin RDS version". Link the relevant ADR or evaluation doc if applicable. -->

## Honest results

<!--
Required for any ML / evaluation change. Report the metric delta on the real eval set,
including when the result is **worse** than the baseline. Negative findings stay in the
record — see `docs/evaluation/baseline-skab.md` for the format.
-->

- Metric:
- Before:
- After:
- Honest take:

## Checklist

- [ ] `make lint` and `make typecheck` pass locally
- [ ] `make test` passes (coverage ≥ 95%)
- [ ] No new long-lived secrets; `gitleaks` pre-commit ran clean
- [ ] Updated docs / ADR / model card if behaviour or interface changed
- [ ] If touching infra: terraform plan reviewed, destructive ops flagged
