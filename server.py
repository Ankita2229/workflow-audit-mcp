"""workflow-audit-mcp — MCP server for auditing ML training experiment specs.

Tools:
  audit(yaml_path)           — run all checks, save report, return findings
  apply_fix(yaml_path, fix_id) — apply a single authorized fix, re-run check
  submit(yaml_path)          — submit to SageMaker (status=ready + clean audit required)
  job_status(job_name)       — check SageMaker job status
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# MCP SDK
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import audit_checks
import fixes
import sagemaker_submit
import sampler
import trajectory_builder

app = Server("workflow-audit-mcp")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="audit",
            description=(
                "Run all statistical pre-flight checks on an experiment YAML and its "
                "trajectory data. Saves report to experiments/<name>.audit.json. "
                "Returns PASS/WARN/FAIL per check with fix proposals for failures."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "yaml_path": {"type": "string", "description": "Path to experiment YAML file"},
                },
                "required": ["yaml_path"],
            },
        ),
        Tool(
            name="apply_fix",
            description=(
                "Apply a single authorized fix by ID. Shows full diff and explanation "
                "before applying. Re-runs the affected check after. Requires prior audit."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "yaml_path": {"type": "string", "description": "Path to experiment YAML file"},
                    "fix_id":    {"type": "string", "description": "Fix ID from audit report (e.g. val_split_type)"},
                },
                "required": ["yaml_path", "fix_id"],
            },
        ),
        Tool(
            name="submit",
            description=(
                "Submit experiment to SageMaker. Blocked if status != ready or any "
                "FAIL check is unresolved. Prints full resolved config and requires "
                "final confirmation before calling SageMaker."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "yaml_path": {"type": "string", "description": "Path to experiment YAML file"},
                    "confirmed": {"type": "boolean", "description": "Set true to confirm submission after reviewing config"},
                },
                "required": ["yaml_path"],
            },
        ),
        Tool(
            name="job_status",
            description="Check SageMaker training job status and recent log lines.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_name": {"type": "string", "description": "SageMaker training job name"},
                },
                "required": ["job_name"],
            },
        ),
        Tool(
            name="sample",
            description=(
                "Sample on-policy trajectories for all on_policy sources in the experiment YAML. "
                "Calls the specified model via OpenRouter for each task × run × iteration, "
                "oracle-scores each completion, and writes episodes to source.path. "
                "Resume-safe: skips (task, run_idx, iteration) triples already written. "
                "Requires OPENROUTER_API_KEY env var. Run this before build_trajectory."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "yaml_path": {"type": "string", "description": "Path to experiment YAML file"},
                },
                "required": ["yaml_path"],
            },
        ),
        Tool(
            name="build_trajectory",
            description=(
                "Mix trajectory sources defined in the YAML into a single JSONL file. "
                "Loads each source (on_policy or existing_jsonl), applies drop_models filters, "
                "samples to the specified fractions without oversampling, sorts by iteration "
                "if requested, and writes to trajectory.path. Run sample() first for on_policy sources."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "yaml_path": {"type": "string", "description": "Path to experiment YAML file"},
                },
                "required": ["yaml_path"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "audit":
            result = audit_checks.run_audit(arguments["yaml_path"])
            return [TextContent(type="text", text=result)]

        elif name == "apply_fix":
            result = fixes.apply_fix(arguments["yaml_path"], arguments["fix_id"])
            return [TextContent(type="text", text=result)]

        elif name == "submit":
            confirmed = arguments.get("confirmed", False)
            result = sagemaker_submit.submit(arguments["yaml_path"], confirmed=confirmed)
            return [TextContent(type="text", text=result)]

        elif name == "job_status":
            result = sagemaker_submit.job_status(arguments["job_name"])
            return [TextContent(type="text", text=result)]

        elif name == "sample":
            result = sampler.sample(arguments["yaml_path"])
            return [TextContent(type="text", text=result)]

        elif name == "build_trajectory":
            result = trajectory_builder.build_trajectory(arguments["yaml_path"])
            return [TextContent(type="text", text=result)]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        import traceback
        return [TextContent(type="text", text=f"ERROR: {e}\n{traceback.format_exc()}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
