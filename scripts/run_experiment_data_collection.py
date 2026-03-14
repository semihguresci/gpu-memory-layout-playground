#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "experiments" / "01_dispatch_basics" / "results" / "tables" / "benchmark_results.json"
COLLECT_SCRIPT = ROOT / "experiments" / "01_dispatch_basics" / "scripts" / "collect_run.py"


def _resolve_binary(explicit_path: str | None) -> Path:
    if explicit_path:
        binary = Path(explicit_path)
        if not binary.is_absolute():
            binary = (ROOT / binary).resolve()
        if not binary.exists():
            raise FileNotFoundError(f"Benchmark binary not found: {binary}")
        return binary

    candidates = [
        ROOT / "build" / "Release" / "gpu_memory_layout_experiments.exe",
        ROOT / "build" / "Debug" / "gpu_memory_layout_experiments.exe",
        ROOT / "build" / "gpu_memory_layout_experiments.exe",
        ROOT / "build" / "gpu_memory_layout_experiments",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find benchmark binary. Build first, for example:\n"
        "  cmake --build build --config Release"
    )


def _run_command(args: list[str]) -> None:
    print(f"[run] {' '.join(args)}")
    subprocess.run(args, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Experiment 01 benchmark and collect fresh run data (no chart generation)."
    )
    parser.add_argument(
        "--binary",
        type=str,
        default=None,
        help="Path to benchmark executable. Defaults to build/Release if available.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Timed iterations for benchmark run.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations for benchmark run.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="4M",
        help="Scratch buffer size for benchmark run (e.g. 4M).",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Enable Vulkan validation layers during benchmark run.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Optional label suffix for collected run filename.",
    )
    parser.add_argument(
        "--no-collect",
        action="store_true",
        help="Skip collecting the run into experiments/01_dispatch_basics/runs.",
    )
    args = parser.parse_args()

    binary = _resolve_binary(args.binary)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    benchmark_cmd = [
        str(binary),
        "--experiment",
        "01_dispatch_basics",
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
        "--size",
        args.size,
        "--output",
        str(DEFAULT_OUTPUT),
    ]
    if args.validation:
        benchmark_cmd.append("--validation")

    _run_command(benchmark_cmd)

    if not args.no_collect:
        collect_cmd = [sys.executable, str(COLLECT_SCRIPT), "--input", str(DEFAULT_OUTPUT)]
        if args.label:
            collect_cmd.extend(["--label", args.label])
        _run_command(collect_cmd)

    print("[ok] Data collection completed.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except subprocess.CalledProcessError as exc:
        print(f"[error] Command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
