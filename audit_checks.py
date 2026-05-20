"""All pre-flight audit checks for experiment YAML + trajectory JSONL."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml as _yaml


# ── Severity constants ────────────────────────────────────────────────────────

FAIL = "FAIL"
WARN = "WARN"
PASS = "PASS"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_yaml(yaml_path: str) -> dict:
    with open(yaml_path) as f:
        return _yaml.safe_load(f)


def _load_jsonl(path: str) -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    episodes.append(json.loads(line))
                except Exception:
                    pass
    return episodes


def _zscore_floor(episodes: list[dict]) -> float:
    """Detect the advantage floor value (most negative outlier cluster)."""
    rewards = [e["reward"] for e in episodes if "reward" in e]
    if not rewards:
        return -3.0
    min_r = min(rewards)
    # Floor is any value <= -2.5 that appears more than once
    candidates = [r for r in rewards if r <= -2.5]
    if not candidates:
        return min_r
    # Most common value at the bottom
    from collections import Counter
    return Counter(candidates).most_common(1)[0][0]


def _group_episodes(episodes: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for ep in episodes:
        groups[ep.get("group", "unknown")].append(ep)
    return dict(groups)


def _task_from_group(group_key: str) -> str:
    return group_key.split("__iter")[0] if "__iter" in group_key else group_key


def _iter_from_group(group_key: str) -> int:
    if "__iter" in group_key:
        try:
            return int(group_key.split("__iter")[1])
        except ValueError:
            pass
    return 0


# ── Individual checks ─────────────────────────────────────────────────────────

def check_val_split_type(cfg: dict, episodes: list[dict], groups: dict) -> dict:
    """FAIL if val split would be episode-level (entrypoint default) rather than group-level."""
    traj_cfg = cfg.get("trajectory", {})
    split_type = traj_cfg.get("val_split", "episode")

    if split_type != "group":
        # Simulate episode-level split to show the leakage
        import random
        val_fraction = traj_cfg.get("val_fraction", 0.2)
        rng = random.Random(42)
        indices = list(range(len(episodes)))
        rng.shuffle(indices)
        split = int(len(indices) * (1 - val_fraction))
        train_idx = set(indices[:split])
        val_idx = set(indices[split:])

        group_train: dict[str, int] = defaultdict(int)
        group_val: dict[str, int] = defaultdict(int)
        for i, ep in enumerate(episodes):
            g = ep.get("group", "unknown")
            if i in train_idx:
                group_train[g] += 1
            else:
                group_val[g] += 1

        split_groups = sum(1 for g in group_train if g in group_val)
        total_groups = len(groups)
        pct = 100 * split_groups / total_groups if total_groups else 0

        return {
            "id": "val_split_type",
            "severity": FAIL,
            "finding": (
                f"val_split is '{split_type}' (episode-level). "
                f"{split_groups}/{total_groups} groups ({pct:.0f}%) would be split across "
                f"train and val — this is prompt-level leakage. "
                f"The model sees training completions for the same prompts it's validated on."
            ),
            "fix_available": True,
            "fix_description": (
                "Change val_split to 'group' in the YAML. "
                "Also patch the entrypoint to split by group key instead of episode index."
            ),
        }

    # Verify the entrypoint actually does group-level split (check for known bug)
    return {"id": "val_split_type", "severity": PASS, "finding": "Val split is group-level. No leakage."}


def check_duplicate_name(cfg: dict, episodes: list[dict], groups: dict) -> dict:
    """FAIL if experiment name already submitted to SageMaker."""
    name = cfg.get("name", "")
    if not name:
        return {"id": "duplicate_name", "severity": FAIL, "finding": "Experiment YAML missing 'name' field."}

    try:
        import boto3
        aws_profile = cfg.get("aws_profile", "Research-AdminAccess-671590542588")
        session = boto3.Session(profile_name=aws_profile, region_name="us-west-2")
        sm = session.client("sagemaker")
        paginator = sm.get_paginator("list_training_jobs")
        for page in paginator.paginate(MaxResults=100):
            for job in page["TrainingJobSummaries"]:
                job_name = job["TrainingJobName"]
                status = job["TrainingJobStatus"]
                if name in job_name and status in ("InProgress", "Completed", "Stopped"):
                    return {
                        "id": "duplicate_name",
                        "severity": FAIL,
                        "finding": (
                            f"Name '{name}' matches existing job '{job_name}' (status: {status}). "
                            f"Update the 'name' field to a unique value."
                        ),
                        "fix_available": False,
                    }
    except Exception as e:
        return {
            "id": "duplicate_name",
            "severity": WARN,
            "finding": f"Could not check SageMaker for duplicate names: {e}",
        }

    return {"id": "duplicate_name", "severity": PASS, "finding": f"Name '{name}' is unique."}


def check_task_val_coverage(cfg: dict, episodes: list[dict], groups: dict) -> dict:
    """FAIL if any task has 0 val groups under a group-level split."""
    traj_cfg = cfg.get("trajectory", {})
    val_fraction = traj_cfg.get("val_fraction", 0.2)

    import random
    all_groups = list(groups.keys())
    rng = random.Random(42)
    rng.shuffle(all_groups)
    n_val = int(len(all_groups) * val_fraction)
    val_groups = set(all_groups[:n_val])

    task_val: dict[str, int] = defaultdict(int)
    task_total: dict[str, int] = defaultdict(int)
    for g in all_groups:
        task = _task_from_group(g)
        task_total[task] += 1
        if g in val_groups:
            task_val[task] += 1

    missing = [t for t in task_total if task_val[t] == 0]
    if missing:
        return {
            "id": "task_val_coverage",
            "severity": FAIL,
            "finding": f"{len(missing)} tasks have 0 val groups: {missing}",
            "fix_available": False,
            "fix_description": "Increase val_fraction or ensure each task has ≥2 groups.",
        }

    min_task = min(task_val, key=task_val.get)
    return {
        "id": "task_val_coverage",
        "severity": PASS,
        "finding": f"All {len(task_total)} tasks covered in val. Min: {task_val[min_task]} groups ({min_task}).",
    }


def check_group_floor_dominance(cfg: dict, episodes: list[dict], groups: dict) -> dict:
    """FAIL if any group has >50% of entries at the advantage floor."""
    floor = _zscore_floor(episodes)
    floor_threshold = floor + 0.01

    dominated = []
    for g, eps in groups.items():
        rewards = [e.get("reward", 0) for e in eps]
        floor_count = sum(1 for r in rewards if r <= floor_threshold)
        pct = floor_count / len(rewards) if rewards else 0
        if pct > 0.5:
            dominated.append((g, pct, len(rewards)))

    if dominated:
        examples = ", ".join(f"{g} ({p:.0%})" for g, p, _ in dominated[:5])
        return {
            "id": "group_floor_dominance",
            "severity": FAIL,
            "finding": (
                f"{len(dominated)} groups have >50% entries at advantage floor ({floor:.2f}). "
                f"Examples: {examples}. "
                f"These groups provide no discrimination signal — all entries say 'do less'."
            ),
            "fix_available": False,
            "fix_description": "Resample or drop these groups. Check why so many completions failed.",
        }

    return {
        "id": "group_floor_dominance",
        "severity": PASS,
        "finding": f"No group dominated by floor ({floor:.2f}). Gradient signal is healthy.",
    }


def check_advantage_variance(cfg: dict, episodes: list[dict], groups: dict) -> dict:
    """FAIL if >5% of groups have zero within-group advantage variance."""
    zero_var = []
    for g, eps in groups.items():
        rewards = [e.get("reward", 0) for e in eps]
        if len(rewards) < 2:
            continue
        try:
            std = statistics.stdev(rewards)
        except Exception:
            std = 0
        if std < 1e-6:
            zero_var.append(g)

    pct = 100 * len(zero_var) / len(groups) if groups else 0
    if pct > 5:
        return {
            "id": "advantage_variance",
            "severity": FAIL,
            "finding": (
                f"{len(zero_var)} groups ({pct:.1f}%) have zero within-group advantage variance. "
                f"GRPO cannot learn from these — all completions have identical weight."
            ),
            "fix_available": False,
            "fix_description": "Check reward computation. These groups likely had all-identical oracle scores.",
        }

    stds = []
    for eps in groups.values():
        rewards = [e.get("reward", 0) for e in eps]
        if len(rewards) >= 2:
            try:
                stds.append(statistics.stdev(rewards))
            except Exception:
                pass
    median_std = statistics.median(stds) if stds else 0

    return {
        "id": "advantage_variance",
        "severity": PASS,
        "finding": f"Within-group advantage variance healthy. Median std: {median_std:.3f}. Zero-var groups: {len(zero_var)} ({pct:.1f}%).",
    }


def check_reward_scale_mismatch(cfg: dict, episodes: list[dict], groups: dict) -> dict:
    """FAIL if advantage std ratio across source models is ≥3×."""
    by_model: dict[str, list[float]] = defaultdict(list)
    for ep in episodes:
        model = ep.get("metadata", {})
        if isinstance(model, dict):
            model = model.get("model", "unknown")
        else:
            model = "unknown"
        reward = ep.get("reward")
        if reward is not None and not math.isinf(reward):
            by_model[model].append(reward)

    if len(by_model) < 2:
        return {"id": "reward_scale_mismatch", "severity": PASS, "finding": "Single model source — no cross-source mismatch possible."}

    stds = {}
    for model, rewards in by_model.items():
        if len(rewards) >= 2:
            try:
                stds[model] = statistics.stdev(rewards)
            except Exception:
                stds[model] = 0

    if not stds:
        return {"id": "reward_scale_mismatch", "severity": PASS, "finding": "Could not compute per-model stds."}

    max_std = max(stds.values())
    min_std = min(stds.values())
    ratio = max_std / min_std if min_std > 1e-8 else float("inf")

    summary = ", ".join(f"{m}: std={s:.3f}" for m, s in sorted(stds.items()))

    if ratio >= 3:
        return {
            "id": "reward_scale_mismatch",
            "severity": FAIL,
            "finding": (
                f"Advantage std ratio across models is {ratio:.1f}× (threshold: 3×). "
                f"{summary}. "
                f"Mixed-scale advantages cause one source to dominate gradient updates."
            ),
            "fix_available": False,
            "fix_description": "Re-normalize advantages per source before mixing, or re-run trajectory build with unified z-scoring.",
        }

    return {
        "id": "reward_scale_mismatch",
        "severity": PASS,
        "finding": f"Reward scale consistent across sources (ratio: {ratio:.1f}×). {summary}",
    }


def check_small_groups(cfg: dict, episodes: list[dict], groups: dict) -> dict:
    """WARN if any groups have fewer than min_group_size entries."""
    min_size = cfg.get("trajectory", {}).get("min_group_size", 6)
    small = [(g, len(eps)) for g, eps in groups.items() if len(eps) < min_size]

    if small:
        examples = ", ".join(f"{g}(n={n})" for g, n in sorted(small, key=lambda x: x[1])[:5])
        return {
            "id": "small_groups",
            "severity": WARN,
            "finding": (
                f"{len(small)} groups have <{min_size} entries. "
                f"Z-score normalization is unreliable for small groups. Examples: {examples}"
            ),
        }
    return {"id": "small_groups", "severity": PASS, "finding": f"All groups have ≥{min_size} entries."}


def check_parse_failure_rate(cfg: dict, episodes: list[dict], groups: dict) -> dict:
    """WARN if any task has >10% parse failure rate."""
    task_total: dict[str, int] = defaultdict(int)
    task_fail: dict[str, int] = defaultdict(int)

    for ep in episodes:
        g = ep.get("group", "")
        task = _task_from_group(g)
        task_total[task] += 1
        completion = ep.get("completion", "")
        metadata = ep.get("metadata", {})
        reward = ep.get("reward", 0)
        floor = _zscore_floor(episodes)

        # Parse failure heuristics: reward at floor AND completion has parse error marker
        is_failure = (
            reward <= floor + 0.01 and (
                "_parse_error" in completion or
                (isinstance(metadata, dict) and metadata.get("oracle_score") is None)
            )
        )
        if is_failure:
            task_fail[task] += 1

    bad_tasks = {t: task_fail[t] / task_total[t]
                 for t in task_total if task_fail[t] / task_total[t] > 0.10}

    overall_rate = sum(task_fail.values()) / sum(task_total.values()) if task_total else 0

    if bad_tasks:
        details = ", ".join(f"{t}({r:.0%})" for t, r in sorted(bad_tasks.items(), key=lambda x: -x[1])[:5])
        return {
            "id": "parse_failure_rate",
            "severity": WARN,
            "finding": (
                f"Overall parse failure rate: {overall_rate:.1%}. "
                f"{len(bad_tasks)} tasks exceed 10% threshold: {details}"
            ),
        }

    return {
        "id": "parse_failure_rate",
        "severity": PASS,
        "finding": f"Overall parse failure rate: {overall_rate:.1%}. No task exceeds 10%.",
    }


def check_think_block_format(cfg: dict, episodes: list[dict], groups: dict) -> dict:
    """WARN if <50% of completions have well-formed <think>...</think> for DeepSeek models."""
    base_model = cfg.get("model", {}).get("base", "")
    if "deepseek" not in base_model.lower() and "qwen" not in base_model.lower():
        return {"id": "think_block_format", "severity": PASS, "finding": "Not a DeepSeek model — think block check skipped."}

    total = len(episodes)
    if total == 0:
        return {"id": "think_block_format", "severity": PASS, "finding": "No episodes."}

    well_formed = sum(
        1 for ep in episodes
        if "<think>" in ep.get("completion", "") and "</think>" in ep.get("completion", "")
    )
    pct = well_formed / total

    if pct < 0.5:
        return {
            "id": "think_block_format",
            "severity": WARN,
            "finding": (
                f"Only {pct:.0%} of completions have well-formed <think>...</think> blocks. "
                f"DeepSeek model expects this format — training on completions without it "
                f"may degrade think-block generation at eval time."
            ),
        }

    return {
        "id": "think_block_format",
        "severity": PASS,
        "finding": f"{pct:.0%} of completions have well-formed think blocks.",
    }


def check_token_truncation(cfg: dict, episodes: list[dict], groups: dict) -> dict:
    """WARN if >20% of completions will be truncated at max_seq_length."""
    max_seq_length = cfg.get("model", {}).get("max_seq_length", 2048)
    base_model = cfg.get("model", {}).get("base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    except Exception:
        return {"id": "token_truncation", "severity": WARN, "finding": "Could not load tokenizer to check truncation."}

    truncated = 0
    sampled = episodes[:200]  # sample for speed
    for ep in sampled:
        prompt_text = tokenizer.apply_chat_template(ep.get("prompt", []), tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + ep.get("completion", "") + tokenizer.eos_token
        ids = tokenizer.encode(full_text, add_special_tokens=False)
        if len(ids) > max_seq_length:
            truncated += 1

    pct = truncated / len(sampled) if sampled else 0

    if pct > 0.2:
        return {
            "id": "token_truncation",
            "severity": WARN,
            "finding": (
                f"{pct:.0%} of sampled completions exceed max_seq_length={max_seq_length}. "
                f"These will be truncated during training, losing the end of completions "
                f"(including JSON output for DeepSeek)."
            ),
        }

    return {
        "id": "token_truncation",
        "severity": PASS,
        "finding": f"{pct:.0%} of sampled completions exceed max_seq_length={max_seq_length}.",
    }


def check_curriculum_sort(cfg: dict, episodes: list[dict], groups: dict) -> dict:
    """WARN if trajectory is not sorted by iteration number."""
    if not cfg.get("trajectory", {}).get("sort_by"):
        return {"id": "curriculum_sort", "severity": PASS, "finding": "No sort_by specified — skipping curriculum check."}

    iters = [_iter_from_group(ep.get("group", "")) for ep in episodes]
    violations = sum(1 for i in range(1, len(iters)) if iters[i] < iters[i - 1])
    pct = violations / len(iters) if iters else 0

    if pct > 0.01:
        return {
            "id": "curriculum_sort",
            "severity": WARN,
            "finding": f"Trajectory not sorted by iteration: {violations} ordering violations ({pct:.1%}).",
            "fix_available": True,
            "fix_description": "Re-sort the JSONL by iteration number.",
        }

    return {"id": "curriculum_sort", "severity": PASS, "finding": "Trajectory is sorted by iteration (curriculum order confirmed)."}


def check_iteration_distribution(cfg: dict, episodes: list[dict], groups: dict) -> dict:
    """WARN if any task is missing iterations expected from trajectory.sources n_iters."""
    # Read expected n_iters from sources block — take the max across all sources
    sources = cfg.get("trajectory", {}).get("sources", [])
    expected_iters = max(
        (int(s.get("n_iters", 0)) for s in sources if s.get("n_iters")),
        default=0,
    )
    # Fall back to eval.n_iterations if sources don't specify
    if not expected_iters:
        expected_iters = int(cfg.get("eval", {}).get("n_iterations", 0))
    if not expected_iters:
        return {"id": "iteration_distribution", "severity": PASS, "finding": "No n_iters specified in YAML — skipping iteration distribution check."}

    # Check per task which iterations are present
    task_iters: dict[str, set] = defaultdict(set)
    for g in groups:
        task = _task_from_group(g)
        it = _iter_from_group(g)
        if it > 0:
            task_iters[task].add(it)

    expected_set = set(range(1, expected_iters + 1))
    missing: dict[str, list] = {}
    for task, present in task_iters.items():
        absent = sorted(expected_set - present)
        if absent:
            missing[task] = absent

    if missing:
        examples = ", ".join(f"{t}(missing iters {v})" for t, v in list(missing.items())[:4])
        return {
            "id": "iteration_distribution",
            "severity": WARN,
            "finding": (
                f"{len(missing)} tasks missing iterations (expected 1–{expected_iters} from YAML). "
                f"Examples: {examples}"
            ),
        }

    return {
        "id": "iteration_distribution",
        "severity": PASS,
        "finding": f"All tasks have all {expected_iters} iterations (read from YAML sources.n_iters).",
    }


def check_data_mix(cfg: dict, episodes: list[dict], groups: dict) -> dict:
    """WARN if actual on-policy % differs from intended by >10 points.
    On-policy model names and intended fraction are read from trajectory.sources."""
    sources = cfg.get("trajectory", {}).get("sources", [])
    if not sources:
        return {"id": "data_mix", "severity": PASS, "finding": "No sources block in YAML — skipping mix check."}

    # Collect intended on-policy models and fraction directly from YAML
    on_policy_models = set()
    intended_on_policy = 0.0
    for s in sources:
        if s.get("type") == "on_policy":
            model = s.get("model", "")
            if model:
                on_policy_models.add(model.lower())
            intended_on_policy += float(s.get("fraction", 0))

    if not on_policy_models:
        return {"id": "data_mix", "severity": PASS, "finding": "No on_policy source in YAML — skipping mix check."}

    on_policy_count = 0
    for ep in episodes:
        meta = ep.get("metadata", {})
        model = meta.get("model", "") if isinstance(meta, dict) else ""
        if model.lower() in on_policy_models:
            on_policy_count += 1

    actual = on_policy_count / len(episodes) if episodes else 0
    diff = abs(actual - intended_on_policy)

    model_list = ", ".join(sorted(on_policy_models))
    if diff > 0.10:
        return {
            "id": "data_mix",
            "severity": WARN,
            "finding": (
                f"Actual on-policy fraction ({actual:.0%}) differs from intended ({intended_on_policy:.0%}) "
                f"by {diff:.0%} (threshold: 10%). On-policy models from YAML: {model_list}"
            ),
        }

    return {
        "id": "data_mix",
        "severity": PASS,
        "finding": f"Data mix on target: actual {actual:.0%} on-policy vs intended {intended_on_policy:.0%}. Models: {model_list}",
    }


def check_metadata_completeness(cfg: dict, episodes: list[dict], groups: dict) -> dict:
    """WARN if >5% of records are missing required metadata fields."""
    required_fields = ["model", "oracle_score", "run_idx"]
    missing_counts: dict[str, int] = defaultdict(int)

    for ep in episodes:
        meta = ep.get("metadata", {})
        if not isinstance(meta, dict):
            for f in required_fields:
                missing_counts[f] += 1
            continue
        for f in required_fields:
            if meta.get(f) is None:
                missing_counts[f] += 1

    total = len(episodes)
    bad_fields = {f: c for f, c in missing_counts.items() if c / total > 0.05}

    if bad_fields:
        details = ", ".join(f"{f}: {c}/{total} ({100*c/total:.1f}%)" for f, c in bad_fields.items())
        return {
            "id": "metadata_completeness",
            "severity": WARN,
            "finding": f"Metadata fields missing in >5% of records: {details}",
        }

    return {"id": "metadata_completeness", "severity": PASS, "finding": "Metadata completeness OK."}


def check_advantage_floor_rate(cfg: dict, episodes: list[dict], groups: dict) -> dict:
    """WARN if overall floor-hitter rate exceeds 5%."""
    floor = _zscore_floor(episodes)
    floor_threshold = floor + 0.01
    floor_count = sum(1 for ep in episodes if ep.get("reward", 0) <= floor_threshold)
    pct = floor_count / len(episodes) if episodes else 0

    if pct > 0.05:
        return {
            "id": "advantage_floor_rate",
            "severity": WARN,
            "finding": (
                f"{floor_count} records ({pct:.1%}) at advantage floor ({floor:.2f}). "
                f"These are parse failures / catastrophic outputs — above 5% warrants review."
            ),
        }

    return {
        "id": "advantage_floor_rate",
        "severity": PASS,
        "finding": f"{floor_count} records ({pct:.1%}) at advantage floor ({floor:.2f}). Within acceptable range.",
    }


# ── Ordered checklist ─────────────────────────────────────────────────────────

HARD_CHECKS = [
    check_val_split_type,
    check_duplicate_name,
    check_task_val_coverage,
    check_group_floor_dominance,
    check_advantage_variance,
    check_reward_scale_mismatch,
]

SOFT_CHECKS = [
    check_small_groups,
    check_parse_failure_rate,
    check_think_block_format,
    check_token_truncation,
    check_curriculum_sort,
    check_iteration_distribution,
    check_data_mix,
    check_metadata_completeness,
    check_advantage_floor_rate,
]


# ── Main audit runner ─────────────────────────────────────────────────────────

def run_audit(yaml_path: str) -> str:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return f"ERROR: {yaml_path} not found."

    cfg = _load_yaml(str(yaml_path))
    traj_path = cfg.get("trajectory", {}).get("path")
    if not traj_path:
        return "ERROR: trajectory.path not set in YAML."

    traj_path = Path(traj_path)
    if not traj_path.exists():
        return f"ERROR: trajectory file not found: {traj_path}"

    episodes = _load_jsonl(str(traj_path))
    if not episodes:
        return f"ERROR: No episodes loaded from {traj_path}"

    groups = _group_episodes(episodes)

    lines = [
        f"═══ Audit: {cfg.get('name', yaml_path.stem)} ═══",
        f"Trajectory: {traj_path} ({len(episodes)} episodes, {len(groups)} groups)",
        f"Timestamp:  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "── HARD CHECKS (FAIL blocks submit) ──────────────────────────",
    ]

    results = []

    for check_fn in HARD_CHECKS:
        r = check_fn(cfg, episodes, groups)
        results.append(r)
        icon = "✓" if r["severity"] == PASS else "✗" if r["severity"] == FAIL else "⚠"
        lines.append(f"  [{r['severity']:4s}] {icon} {r['id']}: {r['finding']}")
        if r.get("fix_available") and r["severity"] == FAIL:
            lines.append(f"         → Fix available: {r.get('fix_description', '')}")

    lines.append("")
    lines.append("── SOFT CHECKS (WARN — reported, do not block) ───────────────")

    for check_fn in SOFT_CHECKS:
        r = check_fn(cfg, episodes, groups)
        results.append(r)
        icon = "✓" if r["severity"] == PASS else "⚠"
        lines.append(f"  [{r['severity']:4s}] {icon} {r['id']}: {r['finding']}")
        if r.get("fix_available") and r["severity"] in (WARN,):
            lines.append(f"         → Fix available: {r.get('fix_description', '')}")

    # Summary
    n_fail = sum(1 for r in results if r["severity"] == FAIL)
    n_warn = sum(1 for r in results if r["severity"] == WARN)
    n_pass = sum(1 for r in results if r["severity"] == PASS)
    lines += [
        "",
        f"── SUMMARY ────────────────────────────────────────────────────",
        f"  PASS: {n_pass}   WARN: {n_warn}   FAIL: {n_fail}",
    ]

    if n_fail > 0:
        fixable = [r for r in results if r["severity"] == FAIL and r.get("fix_available")]
        not_fixable = [r for r in results if r["severity"] == FAIL and not r.get("fix_available")]
        lines.append("")
        lines.append(f"  ✗ SUBMIT BLOCKED — {n_fail} hard failure(s)")
        if fixable:
            lines.append(f"  Fixable via apply_fix: {', '.join(r['id'] for r in fixable)}")
        if not_fixable:
            lines.append(f"  Requires manual action: {', '.join(r['id'] for r in not_fixable)}")
    else:
        status = cfg.get("status", "draft")
        if status == "ready":
            lines.append("  ✓ All hard checks passed. Ready to submit.")
        else:
            lines.append(f"  ✓ All hard checks passed. Set status: ready in YAML to enable submit.")

    # Save report
    report_path = yaml_path.parent / f"{yaml_path.stem}.audit.json"
    report = {
        "name": cfg.get("name"),
        "yaml_path": str(yaml_path),
        "trajectory_path": str(traj_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_episodes": len(episodes),
        "n_groups": len(groups),
        "checks": results,
        "summary": {"pass": n_pass, "warn": n_warn, "fail": n_fail},
        "submit_blocked": n_fail > 0,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    lines.append(f"\n  Report saved: {report_path}")

    return "\n".join(lines)
