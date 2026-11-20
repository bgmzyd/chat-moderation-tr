"""Merge the evaluation runs of two machines into one results table (R4).

Each run of `python -m src.evaluate` writes results/results.json. Copy the run of each machine
into its own folder, then run this script:

  results/server_colab/results.json   (2-vCPU Colab CPU = our "CPU-only server")
  results/laptop_m2/results.json      (Apple M2 MacBook Air, 2 threads)

  python -m src.combine_results   ->  results/results_table.md
"""
import json

from src import config as C

RUNS = [("server_colab", "Server CPU"), ("laptop_m2", "Laptop")]


def main():
    runs = {}
    for folder, label in RUNS:
        path = C.RESULTS_DIR / folder / "results.json"
        if path.exists():
            runs[folder] = json.loads(path.read_text())
    if not runs:
        raise SystemExit("no results found under results/<machine>/results.json")

    first = next(iter(runs.values()))
    by_key = {f: {m["key"]: m for m in r["models"]} for f, r in runs.items()}
    budget = first["latency_budget_ms"]

    lines = [f"Test set: {first['test_set']} (n={first['n_test']}, OFF={first['off_ratio']:.1%}). "
             f"Latency = mean of {first['models'][0]['latency_runs']} single-message runs "
             f"(batch 1, {first['hardware']['torch_threads']} CPU threads); budget {budget:.0f} ms.", ""]
    for f, label in RUNS:
        if f in runs:
            h = runs[f]["hardware"]
            cpu_txt = C.RESULTS_DIR / f / "cpu.txt"          # e.g. `sysctl -n machdep.cpu.brand_string`
            cpu = cpu_txt.read_text().strip() if cpu_txt.exists() else h["cpu"]
            lines.append(f"- {label}: {cpu}, {h['logical_cores']} logical cores, "
                         f"{h['os']}, torch {h['torch']}")
    lines.append("")

    # macro-F1 per machine: INT8 kernels differ slightly between x86 and ARM CPUs
    head, sep = "| Model |", "|---|"
    for f, label in RUNS:
        if f in runs:
            head += f" Macro-F1 ({label}) |"
            sep += "---|"
    head += " Size (MB) |"
    sep += "---|"
    for f, label in RUNS:
        if f in runs:
            head += f" {label} latency ms (p95) | {label} ≤ {budget:.0f} ms |"
            sep += "---|---|"
    lines += [head, sep]

    ref = by_key[next(iter(runs))]
    for key in C.MODEL_KEYS:
        if key not in ref:
            continue
        row = f"| {ref[key]['model']} |"
        for f, _ in RUNS:
            if f in runs:
                r = by_key[f].get(key)
                row += f" {r['macro_f1']:.4f} |" if r else " – |"
        row += f" {ref[key]['size_mb']:.2f} |"
        for f, _ in RUNS:
            if f in runs:
                r = by_key[f].get(key)
                row += (f" {r['latency_mean_ms']:.2f} ({r['latency_p95_ms']:.2f}) |"
                        f" {'yes' if r['within_budget'] else 'no'} |") if r else " – | – |"
        lines.append(row)

    for f, label in RUNS:
        if f in runs:
            for k, v in runs[f].items():
                if k.endswith("prediction_agreement"):
                    lines.append(f"\n{label} – {k.replace('_', ' ')}: {v:.2%}")

    out = C.RESULTS_DIR / "results_table.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
