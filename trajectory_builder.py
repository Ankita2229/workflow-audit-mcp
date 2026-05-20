"""YAML-driven trajectory builder.

Reads trajectory.sources from the experiment YAML, loads/filters each source,
mixes them to the specified fractions, sorts by iteration, and writes the final
JSONL to trajectory.path.

Supported source types:
  on_policy      — pre-sampled file (run sample() first if it doesn't exist)
  existing_jsonl — existing trajectory file with optional model filtering
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import yaml as _yaml


def _evals_root() -> Path:
    return Path(__file__).resolve().parent.parent / "Evals" / "LearningEfficiency"


def _load_jsonl(path: Path) -> list[dict]:
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


def _iter_num(ep: dict) -> int:
    group = ep.get("group", "")
    if "__iter" in group:
        try:
            return int(group.split("__iter")[1])
        except ValueError:
            pass
    return 0


def _model_name(ep: dict) -> str:
    meta = ep.get("metadata", {})
    if isinstance(meta, dict):
        return meta.get("model", "unknown")
    return "unknown"


def build_trajectory(yaml_path: str) -> str:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return f"ERROR: {yaml_path} not found."

    with open(yaml_path) as f:
        cfg = _yaml.safe_load(f)

    traj_cfg = cfg.get("trajectory", {})
    sources = traj_cfg.get("sources", [])
    if not sources:
        return "ERROR: No sources defined in trajectory block."

    out_path_str = traj_cfg.get("path")
    if not out_path_str:
        return "ERROR: trajectory.path not set in YAML."

    evals_root = _evals_root()
    out_path = evals_root / out_path_str

    lines = [f"Building trajectory: {cfg.get('name', yaml_path.stem)}"]

    # ── Load each source ──────────────────────────────────────────────────────
    source_episodes: list[list[dict]] = []
    source_fractions: list[float] = []

    for i, src in enumerate(sources):
        src_type = src.get("type", "")
        src_path_str = src.get("path")
        fraction = float(src.get("fraction", 1.0))

        if src_type == "on_policy":
            if not src_path_str:
                return (
                    f"ERROR: on_policy source {i} has no 'path' field.\n"
                    f"Add path: <output_path> to the source, then run sample('{yaml_path}')."
                )
            src_path = evals_root / src_path_str
            if not src_path.exists():
                return (
                    f"ERROR: on_policy data not found at {src_path}\n"
                    f"Run: sample('{yaml_path}') to generate it first."
                )
            eps = _load_jsonl(src_path)
            lines.append(
                f"  [{i}] on_policy  {src.get('model', '?')}: "
                f"{len(eps)} episodes ← {src_path.name}"
            )

        elif src_type == "existing_jsonl":
            if not src_path_str:
                return f"ERROR: existing_jsonl source {i} missing 'path' field."
            src_path = evals_root / src_path_str
            if not src_path.exists():
                return f"ERROR: source file not found: {src_path}"
            eps = _load_jsonl(src_path)
            drop_models = {m.lower() for m in src.get("drop_models", [])}
            if drop_models:
                before = len(eps)
                eps = [e for e in eps if _model_name(e).lower() not in drop_models]
                lines.append(
                    f"  [{i}] existing   {src_path.name}: "
                    f"{len(eps)} episodes (dropped {before - len(eps)} from {sorted(drop_models)})"
                )
            else:
                lines.append(f"  [{i}] existing   {src_path.name}: {len(eps)} episodes")
        else:
            return f"ERROR: Unknown source type '{src_type}' in source {i}."

        source_episodes.append(eps)
        source_fractions.append(fraction)

    # ── Compute target counts (no source gets oversampled) ────────────────────
    total_fraction = sum(source_fractions)
    norm_fractions = [f / total_fraction for f in source_fractions]

    # Max total we can produce: limited by the source that would need oversampling
    max_total = min(
        len(eps) / frac
        for eps, frac in zip(source_episodes, norm_fractions)
        if frac > 0 and len(eps) > 0
    )

    target_counts = [int(max_total * frac) for frac in norm_fractions]

    # ── Sample from each source ───────────────────────────────────────────────
    rng = random.Random(42)
    mixed: list[dict] = []
    for eps, target in zip(source_episodes, target_counts):
        if len(eps) > target:
            mixed.extend(rng.sample(eps, target))
        else:
            mixed.extend(eps)

    lines.append(
        f"\nMix: {' + '.join(f'{t} ({100*t/len(mixed):.0f}%)' for t in target_counts)} "
        f"= {len(mixed)} total"
    )

    # ── Sort by iteration ─────────────────────────────────────────────────────
    if traj_cfg.get("sort_by") == "iteration":
        mixed.sort(key=_iter_num)
        iters = [_iter_num(ep) for ep in mixed]
        lines.append(f"Sorted by iteration: iter {min(iters)}–{max(iters)}")

    # ── Write output ──────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ep in mixed:
            f.write(json.dumps(ep) + "\n")

    # ── Model breakdown ───────────────────────────────────────────────────────
    model_counts: dict[str, int] = defaultdict(int)
    for ep in mixed:
        model_counts[_model_name(ep)] += 1

    lines.append(f"\nWrote {len(mixed)} episodes → {out_path.name}")
    lines.append("Model breakdown:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {model}: {count} ({100 * count / len(mixed):.1f}%)")

    return "\n".join(lines)
