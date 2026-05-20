"""YAML-driven on-policy sampler.

Reads trajectory.sources[type=on_policy] from the experiment YAML, samples
completions from each source's model via OpenRouter, oracle-scores each one,
and writes episodes to source.path.

Resume-safe: skips (task, run_idx, iteration) triples already written.

Output format per line:
  prompt, completion, reward (0.0 placeholder), group, metadata
  metadata: model, oracle_score, run_idx, iteration, task_name

reward=0.0 is a placeholder — the entrypoint recomputes BSF rewards from
oracle_score + run_idx when recompute_bsf=true in the YAML.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import yaml as _yaml


def _evals_root() -> Path:
    return Path(__file__).resolve().parent.parent / "Evals" / "LearningEfficiency"


def _ensure_evals_path() -> None:
    root = str(_evals_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_done_keys(path: Path) -> set[tuple]:
    """Return set of (task_name, run_idx, iteration) already in output file."""
    done: set[tuple] = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
                meta = ep.get("metadata", {})
                if isinstance(meta, dict):
                    key = (
                        meta.get("task_name", ""),
                        meta.get("run_idx", -1),
                        meta.get("iteration", -1),
                    )
                    done.add(key)
            except Exception:
                pass
    return done


def _parse_params(raw: str) -> dict | None:
    """Parse LLM output, strip non-param keys, return cleaned params dict."""
    try:
        text = raw.strip()
        # Strip markdown fences
        if text.startswith("```"):
            inner = text.split("\n")
            text = "\n".join(inner[1:-1] if inner[-1].strip() in ("```", "") else inner[1:])
            text = text.strip()
        # Unwrap array
        if text.startswith("["):
            arr = json.loads(text)
            obj = arr[0] if (arr and isinstance(arr[0], dict)) else None
        else:
            obj = json.loads(text)
        if not isinstance(obj, dict):
            return None
        return {k: v for k, v in obj.items() if k not in {"rationale", "hypothesis_name", "reasoning"}}
    except Exception:
        return None


def _build_prompt(task, history: list[tuple]) -> list[dict]:
    """Build the structured experiment-log prompt for this iteration."""
    minimize = bool(task.metrics and task.metrics[0].objective == "minimize")
    direction = "minimize" if minimize else "maximize"
    unit = task.target_unit

    if not history:
        user_content = (
            f"Propose an experiment to {direction} "
            f"{task.target_description} ({task.target_unit}).\n"
            "Return ONLY a JSON object with parameter values. "
            "Do not include a rationale."
        )
    else:
        lines = ["Here are your previous experiments and results:"]
        for idx, (params, score) in enumerate(history, 1):
            lines.append(f"  {idx}. {json.dumps(params)} → {score} {unit}")
        scores = [s for _, s in history]
        best = min(scores) if minimize else max(scores)
        lines.append(f"  Best so far: {best} {unit}")
        lines.append("")
        lines.append(
            f"Propose the next experiment to {direction} "
            f"{task.target_description} ({task.target_unit})."
        )
        lines.append(
            "Return ONLY a JSON object with parameter values. "
            "Do not include a rationale."
        )
        user_content = "\n".join(lines)

    return [
        {"role": "system", "content": task.llm_system_prompt},
        {"role": "user", "content": user_content},
    ]


def sample(yaml_path: str) -> str:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return f"ERROR: {yaml_path} not found."

    with open(yaml_path) as f:
        cfg = _yaml.safe_load(f)

    traj_cfg = cfg.get("trajectory", {})
    sources = [s for s in traj_cfg.get("sources", []) if s.get("type") == "on_policy"]

    if not sources:
        return "No on_policy sources — nothing to sample."

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "ERROR: OPENROUTER_API_KEY not set."

    _ensure_evals_path()

    try:
        import openai
    except ImportError:
        return "ERROR: pip install openai"

    try:
        from benchmark.task import get_task
        from benchmark.oracle import load_oracle, predict
        from train.config import TRAIN_BIO_TASKS

        # Register all tasks by importing their configs
        import importlib
        for pkg in ("tasks.bio", "tasks.education"):
            try:
                importlib.import_module(pkg)
            except ImportError:
                pass
    except ImportError as e:
        return f"ERROR importing benchmark modules: {e}"

    evals_root = _evals_root()
    client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    report = [f"Sampling: {cfg.get('name', yaml_path.stem)}"]

    for src_idx, src in enumerate(sources):
        model = src.get("model")
        temperature = float(src.get("temperature", 1.0))
        n_runs = int(src.get("n_runs", 8))
        n_iters = int(src.get("n_iters", 15))
        src_path_str = src.get("path")
        task_names: list[str] = src.get("tasks", TRAIN_BIO_TASKS)

        if not model:
            report.append(f"  [src {src_idx}] ERROR: missing model field.")
            continue
        if not src_path_str:
            report.append(f"  [src {src_idx}] ERROR: missing path field.")
            continue

        out_path = evals_root / src_path_str
        out_path.parent.mkdir(parents=True, exist_ok=True)

        done = _load_done_keys(out_path)
        n_expected = len(task_names) * n_runs * n_iters
        report.append(
            f"\n[src {src_idx}] {model}  temp={temperature}  "
            f"{n_runs} runs × {n_iters} iters × {len(task_names)} tasks = {n_expected} eps"
        )
        report.append(f"  Output: {out_path.name}  ({len(done)} already done)")

        written = skipped = errors = 0

        with open(out_path, "a") as out_f:
            for task_name in task_names:
                try:
                    task = get_task(task_name)
                    oracle_model = load_oracle(task)
                except Exception as e:
                    report.append(f"  SKIP {task_name}: {e}")
                    continue

                minimize = bool(task.metrics and task.metrics[0].objective == "minimize")

                for run_idx in range(n_runs):
                    history: list[tuple[dict, float]] = []

                    for iter_num in range(1, n_iters + 1):
                        key = (task_name, run_idx, iter_num)
                        if key in done:
                            skipped += 1
                            # Rebuild history from the existing episode isn't straightforward,
                            # so we stop replaying at this run and skip to the next.
                            # A fresh run will pick up from iteration 1 on a new run_idx.
                            break

                        prompt = _build_prompt(task, history)

                        for attempt in range(3):
                            try:
                                resp = client.chat.completions.create(
                                    model=model,
                                    temperature=temperature,
                                    max_tokens=512,
                                    messages=prompt,
                                )
                                raw = resp.choices[0].message.content
                                break
                            except Exception as e:
                                if attempt == 2:
                                    report.append(
                                        f"  API error {task_name} run{run_idx} iter{iter_num}: {e}"
                                    )
                                    raw = None
                                else:
                                    time.sleep(5 * (attempt + 1))
                        else:
                            errors += 1
                            continue

                        if raw is None:
                            errors += 1
                            continue

                        params = _parse_params(raw)
                        if params is None:
                            report.append(
                                f"  Parse fail {task_name} run{run_idx} iter{iter_num}: "
                                f"{raw[:60]!r}"
                            )
                            errors += 1
                            continue

                        try:
                            oracle_score = float(predict(oracle_model, params))
                        except Exception:
                            oracle_score = None

                        if oracle_score is not None:
                            history.append((params, oracle_score))

                        ep = {
                            "prompt": prompt,
                            "completion": json.dumps(params),
                            "reward": 0.0,
                            "group": f"{task_name}__iter{iter_num}",
                            "metadata": {
                                "model": model,
                                "oracle_score": oracle_score,
                                "run_idx": run_idx,
                                "iteration": iter_num,
                                "task_name": task_name,
                            },
                        }
                        out_f.write(json.dumps(ep) + "\n")
                        out_f.flush()
                        written += 1

        report.append(f"  written={written}  skipped={skipped}  errors={errors}")

    return "\n".join(report)
