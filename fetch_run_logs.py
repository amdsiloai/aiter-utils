#!/usr/bin/env python3
"""
Fetch Run Logs

Downloads docker_container_*.log artifacts from MLflow child runs
matching experiment and run name filters.

Usage:
    python fetch_run_logs.py --experiment-pattern "amd--Qwen3*" --run-pattern "oob_"
    python fetch_run_logs.py --experiment-pattern "*" --log-pattern "*.log" -o ./server_logs

Environment Variables:
    DATABRICKS_WORKSPACE_URL: Azure Databricks workspace URL
    DATABRICKS_API_KEY: Databricks personal access token
"""

import os
import argparse
import fnmatch
from pathlib import Path

try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

from mlflow_client import DatabricksMLflowClient, format_timestamp


def fetch_logs_for_experiments(
    client: DatabricksMLflowClient,
    experiment_name_pattern: str = "*",
    run_pattern: str = "oob_",
    log_pattern: str = "docker_container_*.log",
    output_dir: str = "./logs",
    verbose: bool = True,
) -> dict:
    """
    Fetch log artifacts from child runs matching filters.

    Args:
        client: DatabricksMLflowClient instance
        experiment_name_pattern: Glob pattern for experiment names
        run_pattern: Pattern to match parent run names
        log_pattern: Glob pattern for artifact filenames to download
        output_dir: Local directory to save logs
        verbose: Print progress

    Returns:
        Summary dict with downloaded/skipped/failed counts
    """
    all_experiments = client.list_experiments()

    if experiment_name_pattern and experiment_name_pattern != "*":
        all_experiments = [
            exp
            for exp in all_experiments
            if fnmatch.fnmatch(exp["name"].lstrip("/"), experiment_name_pattern)
        ]

    if verbose:
        print(f"Found {len(all_experiments)} experiments matching '{experiment_name_pattern}'")
        print(f"Run pattern: {run_pattern}")
        print(f"Log pattern: {log_pattern}")
        print("=" * 80)

    summary = {"downloaded": 0, "skipped": 0, "failed": 0, "errors": []}

    for exp in all_experiments:
        exp_name = exp["name"].lstrip("/")
        exp_id = exp["experiment_id"]

        try:
            parent_runs = client.list_parent_runs(
                experiment_ids=[exp_id], run_name_pattern=run_pattern
            )

            if not parent_runs:
                continue

            if verbose:
                print(f"\nExperiment: {exp_name}")
                print(f"  Found {len(parent_runs)} parent run(s)")

            for parent_run in parent_runs:
                parent_tags = client._extract_tags(parent_run["data"])
                parent_name = parent_tags.get("mlflow.runName", "N/A")
                parent_run_id = parent_run["info"]["run_id"]
                parent_start_time = parent_run["info"].get("start_time", 0)
                timestamp_str = format_timestamp(parent_start_time)

                if verbose:
                    print(f"  Parent: {parent_name} (ID: {parent_run_id[:8]}..., {timestamp_str})")

                child_runs = client.get_child_runs(parent_run_id)

                if not child_runs:
                    if verbose:
                        print(f"    No child runs found, fetching from parent directly")
                    runs_to_process = [parent_run]
                else:
                    if verbose:
                        print(f"    {len(child_runs)} child run(s)")
                    runs_to_process = child_runs

                for run in runs_to_process:
                    child_info = run["info"]
                    child_run_id = child_info["run_id"]
                    child_tags = client._extract_tags(run["data"])
                    child_name = child_tags.get("mlflow.runName", child_run_id[:8])

                    try:
                        artifacts = client.list_artifacts(child_run_id)
                    except Exception as e:
                        if verbose:
                            print(f"    [{child_name}] Error listing artifacts: {e}")
                        summary["failed"] += 1
                        summary["errors"].append(
                            f"{exp_name}/{parent_name}/{child_name}: list_artifacts failed: {e}"
                        )
                        continue

                    matching = [
                        f
                        for f in artifacts
                        if not f.get("is_dir", False)
                        and fnmatch.fnmatch(f.get("path", ""), log_pattern)
                    ]

                    if not matching:
                        summary["skipped"] += 1
                        continue

                    if runs_to_process is child_runs:
                        run_output_dir = (
                            Path(output_dir)
                            / exp_name
                            / f"{parent_name}_{timestamp_str}"
                            / child_run_id[:8]
                        )
                    else:
                        run_output_dir = (
                            Path(output_dir)
                            / exp_name
                            / f"{parent_name}_{timestamp_str}"
                        )
                    run_output_dir.mkdir(parents=True, exist_ok=True)

                    for artifact in matching:
                        artifact_path = artifact["path"]
                        filename = Path(artifact_path).name
                        local_path = run_output_dir / filename

                        if local_path.exists():
                            if verbose:
                                print(f"    [{child_name}] Already exists: {filename}")
                            summary["skipped"] += 1
                            continue

                        try:
                            downloaded_path = client.download_artifact(
                                child_run_id, artifact_path, str(run_output_dir)
                            )
                            summary["downloaded"] += 1

                            if verbose:
                                dl = Path(downloaded_path)
                                size_kb = dl.stat().st_size / 1024 if dl.is_file() else 0
                                print(f"    [{child_name}] Downloaded: {filename} ({size_kb:.1f} KB)")

                        except Exception as e:
                            summary["failed"] += 1
                            summary["errors"].append(
                                f"{exp_name}/{parent_name}/{child_name}/{filename}: {e}"
                            )
                            if verbose:
                                print(f"    [{child_name}] Failed to download {filename}: {e}")

        except Exception as e:
            summary["errors"].append(f"{exp_name}: {e}")
            if verbose:
                print(f"  Error processing experiment: {e}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Fetch docker container logs from MLflow run artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_run_logs.py --experiment-pattern "amd--Qwen3*" --run-pattern "oob_"
  python fetch_run_logs.py --experiment-pattern "*Llama*" --log-pattern "*.log" -o ./server_logs

Environment Variables:
  DATABRICKS_WORKSPACE_URL  Azure Databricks workspace URL
  DATABRICKS_API_KEY        Databricks personal access token
        """,
    )

    parser.add_argument(
        "--experiment-pattern",
        "-e",
        default="*",
        help='Glob pattern for experiment names (default: "*")',
    )
    parser.add_argument(
        "--run-pattern",
        "-r",
        default="oob_",
        help="Pattern to match parent run names (default: oob_)",
    )
    parser.add_argument(
        "--log-pattern",
        "-l",
        default="docker_container_*.log",
        help='Glob pattern for artifact filenames (default: "docker_container_*.log")',
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./logs",
        help="Output directory for downloaded logs (default: ./logs)",
    )
    parser.add_argument(
        "--workspace-url",
        default=None,
        help="Databricks workspace URL (overrides env var)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Databricks API key (overrides env var)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress verbose output"
    )

    args = parser.parse_args()
    verbose = not args.quiet

    workspace_url = args.workspace_url or os.environ.get("DATABRICKS_WORKSPACE_URL")
    api_key = args.api_key or os.environ.get("DATABRICKS_API_KEY")

    if not workspace_url or not api_key:
        print("Error: Databricks credentials required.")
        print("Set DATABRICKS_WORKSPACE_URL and DATABRICKS_API_KEY environment variables,")
        print("or use --workspace-url and --api-key arguments.")
        return 1

    if verbose:
        print("=" * 80)
        print("FETCH RUN LOGS")
        print("=" * 80)
        print(f"Experiment pattern: {args.experiment_pattern}")
        print(f"Run pattern: {args.run_pattern}")
        print(f"Log pattern: {args.log_pattern}")
        print(f"Output directory: {args.output_dir}")
        print("=" * 80)

    client = DatabricksMLflowClient(workspace_url, api_key)

    summary = fetch_logs_for_experiments(
        client=client,
        experiment_name_pattern=args.experiment_pattern,
        run_pattern=args.run_pattern,
        log_pattern=args.log_pattern,
        output_dir=args.output_dir,
        verbose=verbose,
    )

    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"  Downloaded: {summary['downloaded']}")
        print(f"  Skipped:    {summary['skipped']}")
        print(f"  Failed:     {summary['failed']}")

        if summary["errors"]:
            print("\nErrors:")
            for err in summary["errors"]:
                print(f"  - {err}")

    return 0


if __name__ == "__main__":
    exit(main())
