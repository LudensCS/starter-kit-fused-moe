"""
FlashInfer-Bench Profiling Runner.

Runs NCU profiling on the solution.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Solution, TraceSet
from flashinfer_bench.agents import flashinfer_bench_run_ncu
from scripts.pack_solution import pack_solution


def get_trace_set_path() -> str:
    """Get trace set path from environment variable."""
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Please set it to the path of your flashinfer-trace dataset."
        )
    return os.path.abspath(path)


def save_output(output: str, save_to_file: bool = False) -> None:
    """Save or print output based on configuration."""
    if save_to_file:
        # Create outputs directory if it doesn't exist
        output_dir = PROJECT_ROOT / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"profiling_{timestamp}.txt"
        
        # Write to file
        with open(output_file, "w") as f:
            f.write(output)
        print(f"Output saved to: {output_file}")
    else:
        print(output)


def main():
    parser = argparse.ArgumentParser(description="Run NCU profiling on the solution.")
    parser.add_argument(
        "--save-to-file",
        action="store_true",
        help="Save output to outputs folder instead of printing to stdout",
    )
    args = parser.parse_args()
    
    # Set to use only GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "0"
    
    # Pack the solution
    pack_solution()

    # Load solution
    with open(PROJECT_ROOT / "solution.json") as f:
        solution_data = json.load(f)
    solution = Solution(**solution_data)

    # Load trace set and get workload
    trace_set_path = get_trace_set_path()
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    # Use the second workload for profiling (as in original)
    workload = workloads[1].workload

    # Make paths absolute
    for input_name, input_spec in workload.inputs.items():
        if input_spec.type == 'safetensors':
            input_spec.path = os.path.join(trace_set_path, input_spec.path)

    # Run NCU profiling
    result = flashinfer_bench_run_ncu(
        solution=solution,
        workload=workload,
        trace_set_path=trace_set_path,
        #set="detailed",
        sections=[
            "SpeedOfLight",             # 核心利用率 (Compute vs Memory SOL)
            "MemoryWorkloadAnalysis",   # 访存层级分析 (HBM/L2/Shared Memory/TMA)
            "ComputeWorkloadAnalysis",  # 算力流水线分析 (Tensor Core 利用率)
            "SchedulerStats",           # 调度统计 (寻找 Warp Stall 的真凶)
            "InstructionStats",         # 指令分布 (确认是否走 FP8 Tensor Core 路径)
            "WarpStateStats",           # Warp 状态切换 (分析同步/依赖延迟)
        ],
        page="details",
        timeout=3600,
    )
    save_output(result, save_to_file=args.save_to_file)


if __name__ == "__main__":
    main()
