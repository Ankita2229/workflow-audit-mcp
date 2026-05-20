"""SageMaker job submission and status for workflow-audit-mcp."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml as _yaml


def _load_yaml(yaml_path: str) -> dict:
    with open(yaml_path) as f:
        return _yaml.safe_load(f)


def _load_audit_report(yaml_path: Path) -> dict | None:
    report_path = yaml_path.parent / f"{yaml_path.stem}.audit.json"
    if not report_path.exists():
        return None
    with open(report_path) as f:
        return json.load(f)


def submit(yaml_path: str, confirmed: bool = False) -> str:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return f"ERROR: {yaml_path} not found."

    cfg = _load_yaml(str(yaml_path))
    name = cfg.get("name", "")
    status = cfg.get("status", "draft")

    # Gate 1: status must be ready
    if status != "ready":
        return (
            f"BLOCKED: status is '{status}'. Set status: ready in the YAML when you "
            f"have reviewed the config and are ready to submit."
        )

    # Gate 2: audit must have run and passed
    report = _load_audit_report(yaml_path)
    if report is None:
        return (
            "BLOCKED: No audit report found. Run audit first:\n"
            f"  audit('{yaml_path}')"
        )

    if report.get("submit_blocked"):
        fails = [c for c in report.get("checks", []) if c["severity"] == "FAIL"]
        details = "\n".join(f"  ✗ {c['id']}: {c['finding']}" for c in fails)
        return (
            f"BLOCKED: {len(fails)} hard check(s) still failing from last audit:\n"
            f"{details}\n\n"
            f"Fix these issues and re-run audit before submitting."
        )

    # Check audit is not stale (>1 hour old)
    audit_time = datetime.fromisoformat(report["timestamp"])
    age_hours = (datetime.now(timezone.utc) - audit_time).total_seconds() / 3600
    if age_hours > 1:
        return (
            f"BLOCKED: Audit report is {age_hours:.1f}h old. Re-run audit to confirm "
            f"nothing has changed since the last clean pass."
        )

    # Build resolved config summary
    model_cfg = cfg.get("model", {})
    traj_cfg = cfg.get("trajectory", {})
    eval_cfg = cfg.get("eval", {})

    config_summary = [
        "── Resolved Config ─────────────────────────────────────────",
        f"  name:          {name}",
        f"  based_on:      {cfg.get('based_on', 'none')}",
        f"  base_model:    {model_cfg.get('base')}",
        f"  instance:      {model_cfg.get('instance')}",
        f"  epochs:        {model_cfg.get('epochs')}",
        f"  lr:            {model_cfg.get('lr')}",
        f"  lora_rank:     {model_cfg.get('lora_rank')}",
        f"  batch_size:    {model_cfg.get('batch_size')}",
        f"  grad_accum:    {model_cfg.get('grad_accum')}",
        f"  trajectory:    {traj_cfg.get('path')}",
        f"  val_split:     {traj_cfg.get('val_split', 'group')}",
        f"  val_fraction:  {traj_cfg.get('val_fraction', 0.2)}",
        f"  eval_runs:     {eval_cfg.get('n_runs', 4)}",
        f"  eval_iters:    {eval_cfg.get('n_iterations', 15)}",
        f"  eval_shards:   {eval_cfg.get('n_shards', 3)}",
        f"  audit:         CLEAN ({report['summary']['pass']} pass, {report['summary']['warn']} warn)",
        "────────────────────────────────────────────────────────────",
    ]

    if not confirmed:
        return (
            "\n".join(config_summary) + "\n\n"
            "Review the config above. To confirm submission, call:\n"
            f"  submit('{yaml_path}', confirmed=True)"
        )

    # Submit
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Evals" / "LearningEfficiency"))

        from train.infra.deepseek_offline_job import DeepSeekOfflineJobBuilder
        from train.infra.deepseek_eval_job import submit_eval_sharded, _get_clients
        from train.config import GRPOConfig

        aws_profile = cfg.get("aws_profile", "Research-AdminAccess-671590542588")
        traj_path = traj_cfg.get("path")

        grpo_cfg = GRPOConfig(
            name=name,
            description=cfg.get("description", f"Experiment {name}"),
            base_model=model_cfg.get("base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
            lora_rank=int(model_cfg.get("lora_rank", 16)),
            lora_alpha=int(model_cfg.get("lora_rank", 16)) * 2,
            lora_target_modules=("q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
            lora_dropout=0.05,
            trajectory_path=traj_path,
            reward_type="best_so_far",
            val_fraction=float(traj_cfg.get("val_fraction", 0.2)),
            val_seed=42,
            val_split=traj_cfg.get("val_split", "episode"),
            recompute_bsf=bool(cfg.get("recompute_bsf", False)),
            skip_eval=bool(cfg.get("skip_eval", False)),
            group_size=8,
            beta=float(model_cfg.get("beta", 0.1)),
            epochs=int(model_cfg.get("epochs", 2)),
            batch_size=int(model_cfg.get("batch_size", 2)),
            grad_accum=int(model_cfg.get("grad_accum", 8)),
            lr=float(model_cfg.get("lr", 5e-6)),
            warmup_ratio=0.1,
            max_completion_length=int(model_cfg.get("max_seq_length", 2048)),
            eval_n_rollouts=int(eval_cfg.get("n_runs", 4)),
            eval_n_iterations=int(eval_cfg.get("n_iterations", 15)),
            output_dir=f"train/results/{name}",
            bf16=True,
            seed=42,
        )

        builder = DeepSeekOfflineJobBuilder(aws_profile=aws_profile)
        train_job_name = builder.submit(grpo_cfg, trajectory_path=traj_path)

        # Write job name + submitted status back to YAML
        cfg["status"] = "submitted"
        cfg["sagemaker_job"] = train_job_name
        cfg["submitted_at"] = datetime.now(timezone.utc).isoformat()
        with open(yaml_path, "w") as f:
            _yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        return (
            f"✓ Submitted: {train_job_name}\n"
            f"  YAML updated: status=submitted, sagemaker_job={train_job_name}\n\n"
            f"Monitor: job_status('{train_job_name}')\n"
            f"Sharded eval will be submitted automatically after training completes\n"
            f"(run run_offline_v4_pipeline equivalent or poll manually)."
        )

    except Exception as e:
        import traceback
        return f"ERROR during submission:\n{e}\n{traceback.format_exc()}"


def job_status(job_name: str) -> str:
    try:
        import boto3
        from datetime import datetime, timezone

        session = boto3.Session(profile_name="Research-AdminAccess-671590542588", region_name="us-west-2")
        sm = session.client("sagemaker")
        logs = session.client("logs")

        desc = sm.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        secondary = desc.get("SecondaryStatus", "")
        start = desc["TrainingStartTime"]
        elapsed = (datetime.now(timezone.utc) - start).total_seconds() / 3600

        lines = [
            f"Job:      {job_name}",
            f"Status:   {status} / {secondary}",
            f"Elapsed:  {elapsed:.1f}h",
        ]

        # Get recent logs
        log_group = "/aws/sagemaker/TrainingJobs"
        try:
            streams = logs.describe_log_streams(logGroupName=log_group, logStreamNamePrefix=job_name)
            if streams["logStreams"]:
                stream = streams["logStreams"][0]["logStreamName"]
                resp = logs.get_log_events(logGroupName=log_group, logStreamName=stream, limit=15, startFromHead=False)
                lines.append("\nRecent logs:")
                for e in resp["events"]:
                    lines.append(f"  {e['message']}")
        except Exception:
            pass

        return "\n".join(lines)

    except Exception as e:
        return f"ERROR: {e}"
