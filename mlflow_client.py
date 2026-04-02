"""
Lightweight Databricks MLflow client.

Provides DatabricksMLflowClient for listing experiments, searching runs,
downloading artifacts, and basic run introspection.
"""

import os
import time
import fnmatch
from datetime import datetime
from typing import Dict, List, Optional

import requests
import pandas as pd


class DatabricksMLflowClient:
    """Client for interacting with Azure Databricks MLflow API."""

    def __init__(self, workspace_url: str, api_key: str):
        self.workspace_url = workspace_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.mlflow_base_url = f"{self.workspace_url}/api/2.0/mlflow"

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _request_with_retry(
        self, method: str, url: str, max_retries: int = 5, **kwargs
    ) -> requests.Response:
        for attempt in range(max_retries):
            if method.lower() == "get":
                response = requests.get(url, headers=self.headers, **kwargs)
            else:
                response = requests.post(url, headers=self.headers, **kwargs)

            if response.status_code == 429:
                wait_time = min(2**attempt * 5, 60)
                print(
                    f"  Rate limited (429). Waiting {wait_time}s before retry "
                    f"{attempt + 1}/{max_retries}..."
                )
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            return response

        response.raise_for_status()
        return response

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def list_experiments(self, max_results: int = 1000) -> List[Dict]:
        url = f"{self.mlflow_base_url}/experiments/search"
        all_experiments: List[Dict] = []
        page_token = None

        while True:
            params: dict = {"max_results": max_results}
            if page_token:
                params["page_token"] = page_token

            data = self._request_with_retry("get", url, params=params).json()
            all_experiments.extend(data.get("experiments", []))

            page_token = data.get("next_page_token")
            if not page_token:
                break

        return all_experiments

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def search_runs(
        self,
        experiment_ids: Optional[List[str]] = None,
        filter_string: Optional[str] = None,
        run_name: Optional[str] = None,
        max_results: int = 1000,
        order_by: Optional[List[str]] = None,
    ) -> List[Dict]:
        url = f"{self.mlflow_base_url}/runs/search"

        if run_name:
            name_filter = f"tags.mlflow.runName = '{run_name}'"
            filter_string = (
                f"{filter_string} AND {name_filter}" if filter_string else name_filter
            )

        all_runs: List[Dict] = []
        page_token = None

        while True:
            payload: dict = {"max_results": max_results}
            if experiment_ids:
                payload["experiment_ids"] = experiment_ids
            if filter_string:
                payload["filter"] = filter_string
            if order_by:
                payload["order_by"] = order_by
            if page_token:
                payload["page_token"] = page_token

            data = self._request_with_retry("post", url, json=payload).json()
            all_runs.extend(data.get("runs", []))

            page_token = data.get("next_page_token")
            if not page_token:
                break

        return all_runs

    def get_run(self, run_id: str) -> Dict:
        url = f"{self.mlflow_base_url}/runs/get"
        return self._request_with_retry("get", url, params={"run_id": run_id}).json()[
            "run"
        ]

    def list_runs_by_name(
        self,
        experiment_ids: Optional[List[str]] = None,
        run_name_pattern: Optional[str] = None,
        exact_match: bool = False,
        max_results: int = 1000,
    ) -> List[Dict]:
        if not run_name_pattern:
            return self.search_runs(
                experiment_ids=experiment_ids, max_results=max_results
            )

        use_glob = any(c in run_name_pattern for c in ("*", "?", "["))

        # Build server-side LIKE filter to avoid fetching all runs
        if exact_match:
            sql_filter = f"tags.mlflow.runName = '{run_name_pattern}'"
        elif use_glob:
            like_pat = run_name_pattern.replace("*", "%").replace("?", "_")
            sql_filter = f"tags.mlflow.runName LIKE '{like_pat}'"
        else:
            sql_filter = f"tags.mlflow.runName LIKE '%{run_name_pattern}%'"

        runs = self.search_runs(
            experiment_ids=experiment_ids,
            filter_string=sql_filter,
            max_results=max_results,
        )

        # Client-side re-check for glob patterns (LIKE doesn't handle [])
        if use_glob and not exact_match:
            runs = [
                r
                for r in runs
                if fnmatch.fnmatch(
                    self._extract_tags(r["data"]).get("mlflow.runName", ""),
                    run_name_pattern,
                )
            ]

        return runs

    def get_child_runs(
        self, parent_run_id: str, max_results: int = 1000
    ) -> List[Dict]:
        parent_run = self.get_run(parent_run_id)
        experiment_id = parent_run["info"]["experiment_id"]

        return self.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
            max_results=max_results,
        )

    def list_parent_runs(
        self,
        experiment_ids: Optional[List[str]] = None,
        run_name_pattern: Optional[str] = None,
    ) -> List[Dict]:
        all_runs = self.list_runs_by_name(
            experiment_ids=experiment_ids,
            run_name_pattern=run_name_pattern,
            exact_match=False,
        )
        return [
            r
            for r in all_runs
            if "mlflow.parentRunId" not in self._extract_tags(r["data"])
        ]

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def list_artifacts(
        self, run_id: str, path: Optional[str] = None
    ) -> List[Dict]:
        url = f"{self.mlflow_base_url}/artifacts/list"
        all_files: List[Dict] = []
        page_token = None

        while True:
            params: dict = {"run_id": run_id}
            if path:
                params["path"] = path
            if page_token:
                params["page_token"] = page_token

            data = self._request_with_retry("get", url, params=params).json()
            all_files.extend(data.get("files", []))

            page_token = data.get("next_page_token")
            if not page_token:
                break

        return all_files

    def download_artifact(
        self, run_id: str, artifact_path: str, dst_dir: str
    ) -> str:
        import mlflow

        os.environ["DATABRICKS_HOST"] = self.workspace_url
        os.environ["DATABRICKS_TOKEN"] = self.api_key
        mlflow.set_tracking_uri("databricks")

        return mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact_path, dst_path=dst_dir
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_dict_or_list(data: Dict, key: str) -> Dict:
        items = data.get(key, {})
        if isinstance(items, list):
            return {item["key"]: item["value"] for item in items if "key" in item}
        return items

    @staticmethod
    def _extract_tags(run_data: Dict) -> Dict:
        return DatabricksMLflowClient._extract_dict_or_list(run_data, "tags")

    @staticmethod
    def runs_to_dataframe(runs: List[Dict]) -> pd.DataFrame:
        rows = []
        for run in runs:
            info = run["info"]
            run_data = run["data"]
            tags = DatabricksMLflowClient._extract_dict_or_list(run_data, "tags")
            params = DatabricksMLflowClient._extract_dict_or_list(run_data, "params")
            metrics = DatabricksMLflowClient._extract_dict_or_list(run_data, "metrics")

            row: dict = {
                "run_id": info["run_id"],
                "run_name": tags.get("mlflow.runName", "N/A"),
                "status": info["status"],
                "start_time": pd.to_datetime(info["start_time"], unit="ms"),
                "experiment_id": info["experiment_id"],
            }
            if info.get("end_time"):
                row["end_time"] = pd.to_datetime(info["end_time"], unit="ms")

            for k, v in params.items():
                row[f"param_{k}"] = v
            for k, v in metrics.items():
                row[f"metric_{k}"] = v
            for k, v in tags.items():
                if not k.startswith("mlflow."):
                    row[f"tag_{k}"] = v

            rows.append(row)

        return pd.DataFrame(rows)


def format_timestamp(timestamp_ms: int) -> str:
    """Format MLflow timestamp (ms) to YYYYMMDD-HHMMSS."""
    return datetime.fromtimestamp(timestamp_ms / 1000.0).strftime("%Y%m%d-%H%M%S")
