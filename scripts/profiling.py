import os
import sys
import logging
import time
import psutil
import threading
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Solution, TraceSet
from flashinfer_bench.agents import (
    flashinfer_bench_run_ncu,
    flashinfer_bench_list_ncu_options,
)
from scripts.pack_solution import pack_solution

# Enable logging to see underlying debug information
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def monitor_process(interval=300):
    """Monitor system resource usage during profiling."""

    def _monitor():
        while getattr(_monitor, "running", True):
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                gpus = []
                try:
                    import gpustat

                    stats = gpustat.GPUStatCollection.new_query()
                    for gpu in stats:
                        gpus.append(
                            f"GPU{gpu.index}: {gpu.utilization}% util, {gpu.memory_used}/{gpu.memory_total}MB"
                        )
                except:
                    pass

                status = f"[MONITOR] CPU: {cpu_percent}% | RAM: {memory.percent}% ({memory.used // 1024 // 1024}MB/{memory.total // 1024 // 1024}MB)"
                if gpus:
                    status += " | " + " | ".join(gpus)
                print(status)
            except Exception as e:
                pass
            time.sleep(interval)

    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()
    return monitor_thread


def ncu_profile(solution: Solution, workload) -> str:
    """Run ncu profiler on the solution and workload."""
    print("\n" + "=" * 70)
    print("[INFO] Starting NCU profile...")
    print(f"[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        "[WARN] This performs detailed hardware-level profiling and may take ~5 minutes..."
    )
    print("[HINT] Watch the [MONITOR] lines for real-time resource usage")
    print("=" * 70 + "\n")

    trace_set_path = os.environ.get("FIB_DATASET_PATH")
    if not trace_set_path:
        return "ERROR: FIB_DATASET_PATH environment variable not set"

    start_time = time.time()
    print("[PROGRESS] Building command and preparing NCU...\n")

    # Start resource monitoring in background
    monitor_thread = monitor_process(interval=1200)
    try:
        # Reduced timeout from 1800s to 300s for faster iterations
        result = flashinfer_bench_run_ncu(
            solution=solution,
            workload=workload,
            trace_set_path=trace_set_path,
            set="",
            sections=["SpeedOfLight", "MemoryWorkloadAnalysis"],
            page="details",
            timeout=3600,  # Reduced from 1800 seconds to 5 minutes
        )
    finally:
        monitor_thread.running = False

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(
        f"[COMPLETE] NCU profiling finished in {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)"
    )
    print("=" * 70 + "\n")

    # Show first 50 lines if output is long
    lines = result.split("\n")
    if len(lines) > 100:
        print("\n".join(lines[:50]))
        print(
            f"\n... [Output contains {len(lines)} total lines, showing first 50] ...\n"
        )
        print("\n".join(lines[-10:]))
    else:
        print(result)

    return result


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "0"
    print("=" * 70)
    print("FlashInfer-Bench NCU Profiler with Real-time Monitoring")
    print("=" * 70)
    print("\nPacking solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    if "FIB_DATASET_PATH" not in os.environ:
        raise RuntimeError("FIB_DATASET_PATH environment variable not set")

    trace_set = TraceSet.from_path(os.environ["FIB_DATASET_PATH"])
    workload = trace_set.workloads[solution.definition][1].workload

    # print available ncu option sets for debugging
    print("\nNCU options available:\n", flashinfer_bench_list_ncu_options())

    ncu_profile(solution, workload)


if __name__ == "__main__":
    main()
