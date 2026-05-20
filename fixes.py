"""Fix proposal and application for audit failures."""

from __future__ import annotations

import json
from pathlib import Path

import yaml as _yaml

import audit_checks


FIXES: dict[str, dict] = {
    "val_split_type": {
        "description": "Change trajectory.val_split to 'group' in the YAML.",
        "affects_yaml": True,
        "affects_code": False,
    },
    "curriculum_sort": {
        "description": "Re-sort the trajectory JSONL by iteration number.",
        "affects_yaml": False,
        "affects_code": False,
        "affects_data": True,
    },
}


def apply_fix(yaml_path: str, fix_id: str) -> str:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return f"ERROR: {yaml_path} not found."

    with open(yaml_path) as f:
        cfg = _yaml.safe_load(f)

    if fix_id == "val_split_type":
        return _fix_val_split_type(yaml_path, cfg)
    elif fix_id == "curriculum_sort":
        return _fix_curriculum_sort(yaml_path, cfg)
    else:
        return (
            f"No auto-fix available for '{fix_id}'.\n"
            f"Auto-fixable issues: {', '.join(FIXES.keys())}\n"
            f"For '{fix_id}', review the audit finding and apply the change manually."
        )


def _fix_val_split_type(yaml_path: Path, cfg: dict) -> str:
    lines = [
        "FIX: val_split_type",
        "═══════════════════",
        "",
        "PROBLEM:",
        "  val_split is 'episode' (or unset) — episodes from the same group are split",
        "  across train and val, causing prompt-level leakage in the validation signal.",
        "",
        "CHANGE 1 — YAML update:",
        f"  File: {yaml_path}",
        "  Before:  (val_split missing or val_split: episode)",
        "  After:   val_split: group",
        "",
        "CHANGE 2 — Entrypoint patch:",
        "  The entrypoint splits at episode level by default. After updating the YAML,",
        "  the entrypoint must also be patched to read val_split from config and apply",
        "  group-level splitting. The audit will flag if the entrypoint still uses",
        "  episode-level splitting.",
        "",
        "Applying YAML change now...",
    ]

    # Apply to YAML
    if "trajectory" not in cfg:
        cfg["trajectory"] = {}
    cfg["trajectory"]["val_split"] = "group"

    with open(yaml_path, "w") as f:
        _yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    lines.append(f"  ✓ Updated {yaml_path}: trajectory.val_split = group")
    lines.append("")
    lines.append("Re-running check...")

    episodes = audit_checks._load_jsonl(str(cfg["trajectory"]["path"]))
    groups = audit_checks._group_episodes(episodes)
    result = audit_checks.check_val_split_type(cfg, episodes, groups)
    lines.append(f"  Result: [{result['severity']}] {result['finding']}")

    if result["severity"] == audit_checks.PASS:
        lines.append("")
        lines.append("✓ Fix verified. val_split_type now PASSES.")
    else:
        lines.append("")
        lines.append("⚠ YAML updated but check still failing. Entrypoint patch may also be needed.")
        lines.append("  Run: apply_fix('...', 'val_split_entrypoint') when ready.")

    return "\n".join(lines)


def _fix_curriculum_sort(yaml_path: Path, cfg: dict) -> str:
    traj_path = Path(cfg.get("trajectory", {}).get("path", ""))
    if not traj_path.exists():
        return f"ERROR: trajectory file not found: {traj_path}"

    lines = [
        "FIX: curriculum_sort",
        "════════════════════",
        "",
        "PROBLEM:",
        "  Trajectory entries are not sorted by iteration number.",
        "  Curriculum learning (iter1 → iter15) requires ascending order.",
        "",
        f"CHANGE — Re-sort {traj_path} in place:",
        "  Entries will be sorted by iter number extracted from group key.",
        "  Original file will be backed up as <filename>.bak",
        "",
        "Applying sort...",
    ]

    episodes = audit_checks._load_jsonl(str(traj_path))
    backup_path = traj_path.with_suffix(".jsonl.bak")
    import shutil
    shutil.copy(traj_path, backup_path)
    lines.append(f"  ✓ Backup saved: {backup_path}")

    episodes.sort(key=lambda e: audit_checks._iter_from_group(e.get("group", "")))

    with open(traj_path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    lines.append(f"  ✓ Sorted {len(episodes)} episodes by iteration.")

    groups = audit_checks._group_episodes(episodes)
    result = audit_checks.check_curriculum_sort(cfg, episodes, groups)
    lines.append(f"\nRe-check result: [{result['severity']}] {result['finding']}")

    return "\n".join(lines)
