# flashinfer-bench-starter-kit
FlashInfer Bench @ MLSys 2026: Building AI agents to write high performance GPU kernels

## Quick Setup

### Prerequisites

- Python 3.11+
- For local execution: CUDA-capable GPU
- For Modal execution: [Modal](https://modal.com) account

### Installation

```bash
conda create -n fi-bench python=3.12
conda activate fi-bench
pip install flashinfer-bench modal
```

### Download the TraceSet

Clone the competition dataset from HuggingFace:

```bash
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace
```

This contains definitions, workloads, and reference solutions needed for benchmarking.

Set the environment variable to point to the dataset:

```bash
export FIB_DATASET_PATH=/path/to/flashinfer-trace
```

### Modal Setup (for cloud GPU access)

1. Authenticate with Modal:

```bash
modal setup
```

2. Create a Modal Volume and upload the dataset (one-time setup):

```bash
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/flashinfer-trace
```

This uploads the dataset to a persistent Modal Volume for fast access during benchmarks.
