# workflow-audit-mcp Specification

> The key words MUST, MUST NOT, SHOULD, SHOULD NOT, MAY are used per RFC 2119.

## Purpose

An MCP server that audits ML training experiment specs (YAML) against their trajectory
data before allowing submission. It enforces a mandatory statistical pre-flight gate,
presents actionable fix proposals with per-fix user authorization, and submits jobs to
SageMaker only when the spec is explicitly marked ready and all hard checks pass.

## Non-Goals

- MUST NOT auto-submit on any file system event (watch mode, inotify, etc.)
- MUST NOT apply any code or config change without explicit per-fix user authorization
- MUST NOT generate trajectory data (sampling, distillation) — audit only
- MUST NOT manage AWS credentials or IAM roles
- Does not support non-SageMaker backends in v1

## Core Mental Model

An **experiment** is a YAML file in `experiments/` describing a training run: model,
data sources, hyperparams, instance, and a `status` field. The lifecycle is:

```
draft → (audit → fix → audit) → ready → submitted → completed
```

- `draft`: user is still editing. Audit MAY run but submit is blocked.
- `ready`: user has deliberately set this. Submit is allowed only after a clean audit.
- `submitted`: MCP has submitted the job. YAML is frozen — edits require a new name.
- `completed`: eval results are available.

A **check** is a single falsifiable assertion about the experiment spec or its trajectory
data. Each check has a severity: FAIL (blocks submit), WARN (reported, does not block).

A **fix** is a proposed code or config change returned by a failing check. The MCP
MUST present the full diff and a plain-English explanation before applying. The user
MUST authorize each fix individually by name. The MCP MUST re-run the affected check
after applying and confirm it now passes.

## YAML Schema Contract

Every experiment YAML MUST contain:

```yaml
name: <string>          # unique identifier — MCP rejects duplicates against submitted jobs
status: draft | ready   # user sets to "ready" when done editing
based_on: <string>      # optional — name of parent experiment (provenance)

model:
  base: <hf model id>
  instance: <sagemaker instance type>
  epochs: <int>
  lr: <float>
  lora_rank: <int>
  batch_size: <int>
  grad_accum: <int>

trajectory:
  path: <local jsonl path>   # OR sources block for on-the-fly build
  val_split: group           # MUST be "group" — episode-level is rejected (FAIL)
  val_fraction: <float>
  min_group_size: <int>      # groups below this are flagged (WARN if >0, FAIL if >5%)

eval:
  n_runs: <int>
  n_iterations: <int>
  n_shards: <int>
```

## Behavioral Invariants

### Audit

- The MCP MUST run all checks in the checklist before returning any result.
- Checks MUST be deterministic given the same YAML and JSONL.
- The MCP MUST return a structured report: one entry per check with severity, finding,
  and fix proposal (if applicable).
- The MCP MUST save the audit report to `experiments/<name>.audit.json` alongside the YAML.
- A second audit run on the same YAML+JSONL MUST produce the same severities (idempotent).
- If any FAIL check has no fix proposal, the MCP MUST explain why it cannot auto-fix
  and what manual action is required.

### Submit

- The MCP MUST NOT submit a job if `status != "ready"`.
- The MCP MUST NOT submit a job if any FAIL check remains unresolved in the audit report.
- The MCP MUST NOT submit a job if an experiment with the same `name` already exists in
  SageMaker with status InProgress, Completed, or Stopped.
- Before submitting, the MCP MUST print the full resolved config and ask for final
  confirmation.
- After submitting, the MCP MUST write `status: submitted` and the SageMaker job name
  back into the YAML.

### Fix Authorization

- The MCP MUST present: (1) what is wrong, (2) exactly what will change (full diff),
  (3) why this fixes it.
- The MCP MUST wait for explicit authorization before applying any change.
- The MCP MUST NOT batch-apply multiple fixes in a single authorization.
- After applying a fix, the MCP MUST re-run the affected check and report pass/fail.

## Audit Checklist

### Hard Checks (FAIL — block submit)

| ID | Check | Threshold |
|----|-------|-----------|
| `val_split_type` | Val split is group-level, not episode-level | Any episode-level split |
| `duplicate_name` | Experiment name not already submitted | Any duplicate |
| `task_val_coverage` | Every task has ≥1 group in val | Any task with 0 val groups |
| `group_floor_dominance` | No group has >50% of entries at advantage floor | >50% floor in any group |
| `advantage_variance` | Within-group advantage std > 0 for >95% of groups | >5% zero-variance groups |
| `reward_scale_mismatch` | Advantage std ratio across sources <3× | Ratio ≥3× |

### Soft Checks (WARN — reported, do not block)

| ID | Check | Threshold |
|----|-------|-----------|
| `small_groups` | Groups with <6 entries flagged | Any |
| `parse_failure_rate` | Per-task parse failure rate | >10% any task |
| `parse_failure_trend` | Failure rate should decrease iter1→iter15 | Increasing trend |
| `think_block_format` | % completions with well-formed `<think>...</think>` | <50% for DeepSeek |
| `token_truncation` | % completions truncated at max_seq_length | >20% |
| `curriculum_sort` | Entries sorted by iteration number | Out of order |
| `data_mix` | Actual on-policy % within 10% of intended | >10% off |
| `metadata_completeness` | Required fields present per source model | >5% missing any field |
| `advantage_floor_rate` | Overall % of entries at −3.0 floor | >5% |
| `iteration_distribution` | All 15 iterations present per task | Any task missing iters |

## Known Limitations / Accepted Trade-offs

- Audit checks are static (data at rest). They do not simulate training dynamics.
- Token truncation check uses local tokenizer; if tokenizer version differs from
  container, counts may be slightly off.
- `duplicate_name` check queries SageMaker list API — requires AWS credentials at
  audit time.
- Fix proposals are generated for structural issues only. Domain-level concerns
  (e.g., "is this reward signal meaningful for this task?") are reported as WARNs
  with no auto-fix.

## Open Questions

None — resolved before implementation.

## Verifiable Conditions

- Given a JSONL with 94% groups split across train/val, `val_split_type` MUST return FAIL.
- Given a YAML with `status: draft`, `submit` MUST return an error without calling SageMaker.
- Given a YAML with `name` matching an InProgress SageMaker job, `submit` MUST return FAIL.
- Given a JSONL where all entries in a group share the same advantage value, `advantage_variance` MUST return FAIL for that group.
- After `apply_fix: val_split_type`, a re-audit MUST return PASS for `val_split_type`.
- The audit report at `experiments/<name>.audit.json` MUST exist after any audit run.
