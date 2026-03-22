# llama-benchy 🦙📊

`llama-benchy` is a simple yet powerful benchmarking and visualization tool for `llama.cpp` models. It automates the process of running `llama-bench` across multiple models and context depths, storing results in a YAML database, and generating comparative line graphs.

## Features

- **Automated Benchmarking**: Scans your model cache for `.gguf` files and benchmarks them using `llama-bench`.
- **Flexible Context Depths**: Test models across multiple context lengths (defined in `config.yaml`).
- **YAML Database**: Stores results in a structured YAML format for easy versioning and analysis.
- **Visual Analytics**: Generates professional charts for:
  - **Prompt Processing (pp512)**: Tokens per second for initial prompt ingestion.
  - **Token Generation (tg128)**: Tokens per second for sequential text generation.
- **Multi-DB Comparison**: Compare benchmark results from different systems (e.g., ROCm vs CUDA) by passing multiple database files to the graphing command.
- **Smart Filtering**: Exclude specific patterns (like multimodal `mmproj` or specific model architectures) automatically.

## Requirements

- Python 3.x
- `llama-bench` (must be in your `PATH`, part of the `llama.cpp` repository)
- Python packages:
  - `matplotlib`
  - `pyyaml`

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/nlbutts/llama-benchy
    cd llama-benchy
    ```

2.  **Install dependencies**:
    ```bash
    pip install matplotlib pyyaml
    ```

## Configuration

Edit `config.yaml` to specify your model cache and benchmarking parameters:

```yaml
model_cache: ~/.cache/llama.cpp  # Where your .gguf models are stored

llama_bench:
  flash_attn: 1      # -fa flag
  threads: 8         # -t 8
  batch_size: 2048   # -b 2048
  ubatch_size: 512   # -ub 512
  n_gpu_layers: 99   # -ngl 99
  repetitions: 1     # -r 1
  no_warmup: false   # --no-warmup

depths:              # -d list of context depths
  - 0
  - 4096
  - 16384
  - 32768
  - 131072

excluded_patterns:  # Patterns to ignore when searching for models
  - mmproj
  - moe
```

## Usage

### 1. Run Benchmarks
Run the `benchmark` mode to start testing all models in your cache:
```bash
python llama-benchy.py benchmark
```
*Note: This will save/append results to `benchmarks.yaml` by default.*

### 2. Generate Graphs
To create PNG charts for all benchmarked models:
```bash
python llama-benchy.py graph
```
*The graphs will be saved in the `graphs/` directory.*

### Advanced: Comparing Different Systems
If you have benchmarked models on different hardware and saved them to different files (e.g., `results_cuda.yaml` and `results_rocm.yaml`), you can compare them on one graph:
```bash
python llama-benchy.py graph --db results_cuda.yaml results_rocm.yaml
```

## Output Examples

Graphs show `Tokens/sec` on the Y-axis and `Context Depth` on the X-axis, with separate charts for Prompt Processing and Token Generation.

![Example Graph](graphs/Example_Model.png) *(Note: Placeholder reference to the generated files in the graphs/ folder)*

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 Nicholas Butts
