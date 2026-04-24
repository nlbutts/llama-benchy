#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime, timezone
from glob import glob
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def expand_path(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def get_model_name(path: str) -> str:
    """Extract a descriptive model name, including repo if it's a HF layout."""
    parts = Path(path).parts
    for part in parts:
        if part.startswith("models--"):
            repo = part.replace("models--", "").replace("--", "/")
            return f"{repo}/{Path(path).stem}"
    return Path(path).stem


def find_models(cache_dir: str, excluded_patterns: list[str]) -> list[str]:
    cache_dir = expand_path(cache_dir)
    # Recursively find models to support HuggingFace cache structure
    patterns = ["**/*.gguf", "**/*.GGUF"]
    models = []
    for pat in patterns:
        for f in glob(os.path.join(cache_dir, pat), recursive=True):
            skip = False
            for ex in excluded_patterns:
                if ex.lower() in f.lower():
                    skip = True
                    break
            if not skip:
                models.append(f)
    return sorted(list(set(models)))


LONG_TO_SHORT = {
    "batch_size": "-b",
    "ubatch_size": "-ub",
    "n_gpu_layers": "-ngl",
    "threads": "-t",
    "repetitions": "-r",
    "flash_attn": "-fa",
    "no_warmup": "--no-warmup",
}


def build_llama_bench_cmd(
    model_path: str, config: dict, depths: list[int]
) -> list[str]:
    llm = config.get("llama_bench", {})
    cmd = [
        "llama-bench",
        "-m",
        model_path,
        "-d",
        ",".join(map(str, depths)),
        "-o",
        "csv",
    ]
    for k, v in llm.items():
        flag = LONG_TO_SHORT.get(k, f"-{k}")
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.append(flag)
            cmd.append(str(v))
    return cmd


def parse_csv_result(csv_text: str, model_path: str, depth: int) -> list[dict]:
    lines = csv_text.strip().split("\n")
    header = None
    results = []
    for line in lines:
        if line.startswith("WARNING:") or line.startswith("ggml_") or not line.strip():
            continue
        if line.startswith("build_"):
            header = line
            continue
        if header is None:
            continue
        reader = csv.DictReader(StringIO(header + "\n" + line))
        for row in reader:
            n_prompt = int(row.get("n_prompt", 0))
            n_gen = int(row.get("n_gen", 0))
            avg_ts = float(row.get("avg_ts", 0))
            stddev_ts = float(row.get("stddev_ts", 0))
            results.append(
                {
                    "model_path": row.get("model_filename", model_path),
                    "model_name": get_model_name(model_path),
                    "model_size": int(row.get("model_size", 0)),
                    "n_params": int(row.get("model_n_params", 0)),
                    "cpu_info": row.get("cpu_info", ""),
                    "gpu_info": row.get("gpu_info", ""),
                    "backend": row.get("backends", ""),
                    "n_prompt": n_prompt,
                    "n_gen": n_gen,
                    "depth": int(row.get("n_depth", depth)),
                    "test_time": row.get("test_time", ""),
                    "avg_ns": float(row.get("avg_ns", 0)),
                    "stddev_ns": float(row.get("stddev_ns", 0)),
                    "avg_ts": avg_ts,
                    "stddev_ts": stddev_ts,
                }
            )
    return results


def run_benchmark(config: dict, db_path: str):
    cache_dir = config.get("model_cache", "~/.cache/huggingface/hub")
    excluded = config.get("excluded_patterns", [])
    depths = config.get("depths", [0, 4096, 16384])

    models = find_models(cache_dir, excluded)
    if not models:
        print(f"No models found in {cache_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(models)} models")

    db_data = {"benchmarks": [], "generated_at": datetime.now(timezone.utc).isoformat()}
    if os.path.exists(db_path):
        with open(db_path, "r") as f:
            db_data = yaml.safe_load(f)
        if "benchmarks" not in db_data:
            db_data["benchmarks"] = []
    else:
        db_data = {
            "benchmarks": [],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    for model in models:
        model_display_name = get_model_name(model)
        print(f"\nBenchmarking: {model_display_name}")
        cmd = build_llama_bench_cmd(model, config, depths)
        print(f"  Command: {' '.join(cmd)}")
        try:
            # Find the absolute path of the llama-bench executable
            llama_bench_path = subprocess.getoutput("which llama-bench")
            print(f"  llama-bench path: {llama_bench_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            rows = parse_csv_result(result.stdout, model, 0)
            if rows:
                first = rows[0]
                model_entry = {
                    "model_name": get_model_name(first["model_path"]),
                    "model_path": first["model_path"],
                    "model_size": first["model_size"],
                    "n_params": first["n_params"],
                    "cpu_info": first["cpu_info"],
                    "gpu_info": first["gpu_info"],
                    "backend": first["backend"],
                    "results": [],
                }
                for row in rows:
                    model_entry["results"].append(
                        {
                            "n_prompt": row["n_prompt"],
                            "n_gen": row["n_gen"],
                            "depth": row["depth"],
                            "avg_ts": row["avg_ts"],
                            "stddev_ts": row["stddev_ts"],
                        }
                    )
                db_data["benchmarks"].append(model_entry)
                print(f"    -> got {len(rows)} result rows")
            else:
                print(f"    -> no results")
        except subprocess.CalledProcessError as e:
            print(f"    -> error: {e.stderr.strip().split(chr(10))[-1]}")
            continue

    db_data["generated_at"] = datetime.now(timezone.utc).isoformat()
    with open(db_path, "w") as f:
        yaml.dump(db_data, f, default_flow_style=False, sort_keys=False)
    print(f"\nResults saved to {db_path}")


def generate_graphs(db_paths: list[str], output_dir: str):
    """Graph mode: compare different YAML files (benchmark runs) per model.

    For each model that appears in multiple YAML files, creates a graph
    with one line per YAML file, showing how performance changed across runs.
    """
    # all_data[model_name][mode][db_label] = [(depth, ts, std), ...]
    all_data: dict[str, dict[str, dict[str, list[tuple]]]] = {}

    for path in db_paths:
        if not os.path.exists(path):
            print(f"Warning: {path} not found", file=sys.stderr)
            continue

        db_label = Path(path).stem
        with open(path, "r") as f:
            db_data = yaml.safe_load(f)

        benchmarks = db_data.get("benchmarks", [])
        for b in benchmarks:
            path_val = b.get("model_path", "Unknown")
            model_name = (
                Path(path_val).stem
                if path_val != "Unknown"
                else b.get("model_name", "Unknown")
            )

            all_data.setdefault(model_name, {"pp": {}, "tg": {}})

            rows = []
            if "results" in b:
                rows = b["results"]
            else:
                rows = [b]

            for r in rows:
                if r.get("n_prompt") == 512 and r.get("n_gen") == 0:
                    all_data[model_name]["pp"].setdefault(db_label, []).append(
                        (r["depth"], r["avg_ts"], r["stddev_ts"])
                    )
                elif r.get("n_prompt") == 0 and r.get("n_gen") == 128:
                    all_data[model_name]["tg"].setdefault(db_label, []).append(
                        (r["depth"], r["avg_ts"], r["stddev_ts"])
                    )

    if not all_data:
        print("No benchmark data found", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    markers = ["o", "s", "^", "v", "<", ">", "D", "p", "*", "h"]

    for model_name, modes in sorted(all_data.items()):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(model_name, fontsize=11, fontweight="bold")

        for ax, mode_key in [(ax1, "pp"), (ax2, "tg")]:
            series = modes[mode_key]
            title = "Prompt Processing (pp512)" if mode_key == "pp" else "Token Generation (tg128)"
            ax.set_title(title)

            if series:
                all_depths = sorted(
                    list(set(d for rows in series.values() for d, _, _ in rows))
                )
                depth_to_idx = {d: i for i, d in enumerate(all_depths)}

                for i, (db_label, rows) in enumerate(sorted(series.items())):
                    rows.sort(key=lambda x: x[0])
                    x_vals = [depth_to_idx[r[0]] for r in rows]
                    ts = [r[1] for r in rows]
                    std = [r[2] for r in rows]
                    ax.errorbar(
                        x_vals, ts, yerr=std,
                        fmt=f"-{markers[i % len(markers)]}",
                        label=db_label, capsize=5, lw=2, markersize=6,
                    )

                ax.set_xticks(range(len(all_depths)))
                ax.set_xticklabels([str(d) for d in all_depths])
                ax.set_xlabel("Depth")
                ax.set_ylabel("Tokens/sec")
                ax.grid(True, linestyle="--", alpha=0.6)
                if len(series) > 1:
                    ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, f"No {mode_key} data", ha="center", va="center", transform=ax.transAxes)

        safe_name = model_name.replace("/", "_").replace(" ", "_")
        fig_path = os.path.join(output_dir, f"{safe_name}.png")
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved: {fig_path}")


def compare_models(db_path: str, output_dir: str):
    """Compare mode: single YAML file, plot all models in one graph.

    Creates a single graph with two subplots (pp and tg), where each line
    represents a different model from the same benchmark run.
    """
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(db_path, "r") as f:
        db_data = yaml.safe_load(f)

    benchmarks = db_data.get("benchmarks", [])
    if not benchmarks:
        print("No benchmark data found", file=sys.stderr)
        sys.exit(1)

    # model_display_name[mode] = [(depth, ts, std), ...]
    model_data: dict[str, dict[str, list[tuple]]] = {}

    for b in benchmarks:
        model_name = b.get("model_name", "Unknown")
        model_data.setdefault(model_name, {"pp": [], "tg": []})

        rows = b.get("results", [b])
        for r in rows:
            if r.get("n_prompt") == 512 and r.get("n_gen") == 0:
                model_data[model_name]["pp"].append(
                    (r["depth"], r["avg_ts"], r["stddev_ts"])
                )
            elif r.get("n_prompt") == 0 and r.get("n_gen") == 128:
                model_data[model_name]["tg"].append(
                    (r["depth"], r["avg_ts"], r["stddev_ts"])
                )

    os.makedirs(output_dir, exist_ok=True)
    markers = ["o", "s", "^", "v", "<", ">", "D", "p", "*", "h"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    run_label = Path(db_path).stem
    fig.suptitle(f"Model Comparison: {run_label}", fontsize=13, fontweight="bold")

    for ax, mode_key in [(ax1, "pp"), (ax2, "tg")]:
        title = "Prompt Processing (pp512)" if mode_key == "pp" else "Token Generation (tg128)"
        ax.set_title(title)

        all_depths = sorted(
            list(set(d for name in model_data for d, _, _ in model_data[name][mode_key]))
        )
        depth_to_idx = {d: i for i, d in enumerate(all_depths)}

        for i, (model_name, modes) in enumerate(sorted(model_data.items())):
            rows = sorted(modes[mode_key], key=lambda x: x[0])
            if not rows:
                continue

            x_vals = [depth_to_idx[r[0]] for r in rows]
            ts = [r[1] for r in rows]
            std = [r[2] for r in rows]

            short_name = model_name.split("/")[-1] if "/" in model_name else model_name
            ax.errorbar(
                x_vals, ts, yerr=std,
                fmt=f"-{markers[i % len(markers)]}",
                label=short_name, capsize=5, lw=2, markersize=6,
            )

        ax.set_xticks(range(len(all_depths)))
        ax.set_xticklabels([str(d) for d in all_depths], rotation=45, ha="right")
        ax.set_xlabel("Depth")
        ax.set_ylabel("Tokens/sec")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=8, loc="upper right")

    fig_path = os.path.join(output_dir, f"{run_label}_compare.png")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="llama-benchy - Benchmark and graph llama.cpp models"
    )
    parser.add_argument(
        "mode",
        choices=["benchmark", "graph", "compare"],
        help="Mode: benchmark (run llama-bench), graph (compare runs per model), or compare (all models in one run)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--db",
        default=["benchmarks.yaml"],
        nargs="+",
        help="Path to benchmark database YAML file(s) (default: benchmarks.yaml)",
    )
    parser.add_argument(
        "--output",
        default="graphs",
        help="Output directory for graphs (default: graphs)",
    )
    args = parser.parse_args()

    if args.mode == "benchmark":
        config = load_config(args.config)
        run_benchmark(config, args.db[0])
    elif args.mode == "graph":
        generate_graphs(args.db, args.output)
    elif args.mode == "compare":
        compare_models(args.db[0], args.output)


if __name__ == "__main__":
    main()
