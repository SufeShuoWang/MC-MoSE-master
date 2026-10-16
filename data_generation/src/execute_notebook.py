from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import psutil

from src.path_utils import find_project_root
from src.atomic_write import write_json_atomic
from src.runtime_paths import (
    clean_roots,
    execution_context,
    run_log_root,
)


def execute_notebook(
    path: str | Path,
    *,
    run_mode: str,
    timeout: int,
    kernel_name: str | None = None,
) -> Path:
    root = find_project_root(Path(path).resolve())
    target = Path(path).resolve()
    notebook = nbformat.read(target, as_version=4)
    old_mode = os.environ.get("KANIEL_RUN_MODE")
    old_name = os.environ.get("KANIEL_NOTEBOOK_NAME")
    os.environ["KANIEL_RUN_MODE"] = run_mode
    os.environ["KANIEL_NOTEBOOK_NAME"] = target.stem
    context = execution_context()
    started = time.perf_counter()
    peak_rss = 0
    stop_monitor = threading.Event()

    def monitor_memory() -> None:
        nonlocal peak_rss
        process = psutil.Process()
        while not stop_monitor.wait(0.25):
            try:
                values = [process, *process.children(recursive=True)]
                peak_rss = max(
                    peak_rss,
                    sum(value.memory_info().rss for value in values if value.is_running()),
                )
            except (psutil.Error, OSError):
                continue

    monitor = threading.Thread(target=monitor_memory, daemon=True)
    monitor.start()
    try:
        client = NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name=kernel_name or notebook.metadata.kernelspec.name,
            resources={"metadata": {"path": str(root)}},
            allow_errors=False,
        )
        client.execute(cwd=str(root))
    finally:
        stop_monitor.set()
        monitor.join(timeout=2)
        if old_mode is None:
            os.environ.pop("KANIEL_RUN_MODE", None)
        else:
            os.environ["KANIEL_RUN_MODE"] = old_mode
        if old_name is None:
            os.environ.pop("KANIEL_NOTEBOOK_NAME", None)
        else:
            os.environ["KANIEL_NOTEBOOK_NAME"] = old_name
    if context == "clean_rebuild_compute":
        executed_dir = clean_roots(root)["outputs"] / "executed_notebooks"
        executed_dir.mkdir(parents=True, exist_ok=True)
        output_target = executed_dir / target.name
    else:
        output_target = target
    temporary = output_target.with_suffix(output_target.suffix + ".tmp")
    nbformat.write(notebook, temporary)
    temporary.replace(output_target)
    summary_path = run_log_root(root) / f"{target.stem}_run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "orchestrator_elapsed_seconds": round(time.perf_counter() - started, 6),
                "peak_process_tree_rss_bytes": int(peak_rss),
                "executed_notebook_path": str(output_target.relative_to(root)),
                "executed_kernel_name": (
                    kernel_name or notebook.metadata.kernelspec.name
                ),
            }
        )
        write_json_atomic(summary_path, summary)
    if run_mode == "full":
        checkpoint_path = run_log_root(root) / "full_run_checkpoint.json"
        checkpoint = (
            json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint_path.exists()
            else {"run_mode": "full", "completed": {}}
        )
        name = target.stem
        checkpoint["completed"][name] = {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "summary_path": str(summary_path.relative_to(root)),
            "summary_exists": summary_path.exists(),
        }
        checkpoint["last_completed"] = name
        write_json_atomic(checkpoint_path, checkpoint)
    return output_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="+")
    parser.add_argument("--run-mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--kernel-name")
    args = parser.parse_args()
    for value in args.notebooks:
        output = execute_notebook(
            value,
            run_mode=args.run_mode,
            timeout=args.timeout,
            kernel_name=args.kernel_name,
        )
        print(output)


if __name__ == "__main__":
    main()
