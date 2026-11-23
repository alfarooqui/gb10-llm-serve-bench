#!/usr/bin/env python3
"""Merge summary.json files from several runs into one comparison table.

Usage:
  python scripts/04_compare.py results/gb10-128g/<stamp> results/rtx-pro-6000-96g/<stamp>
Writes results/comparison.md (or --out).
Only levels present in every run are compared, and only if the model and
benchmark settings match. Mismatches are reported, not hidden.
"""
import argparse
import json
import pathlib


def load(run_dir):
    p = pathlib.Path(run_dir) / "summary.json"
    data = json.loads(p.read_text())
    return data["meta"], {lvl["concurrency"]: lvl for lvl in data["levels"]}, p.parent.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="result folders containing summary.json")
    ap.add_argument("--out", default="results/comparison.md")
    args = ap.parse_args()

    loaded = [load(r) for r in args.runs]
    metas = [m for m, _, _ in loaded]

    # Fairness check: the same model and benchmark settings across runs.
    keys = ("model", "max_tokens", "temperature", "prompts")
    mismatches = [k for k in keys if len({str(m.get(k)) for m in metas}) > 1]

    labels = [f"{m['profile']} ({stamp})" for m, (_, _, stamp) in zip(metas, loaded)]
    common = set.intersection(*[set(levels) for _, levels, _ in loaded])

    lines = ["# Cross-platform comparison", ""]
    if mismatches:
        lines += [f"**WARNING: settings differ across runs: {', '.join(mismatches)}. "
                  "Comparison is not fair.**", ""]
    lines += [f"- Model: `{metas[0]['model']}`" if not mismatches or 'model' not in mismatches
              else "- Model: mixed", ""]

    for metric, title, fmt in [
        ("aggregate_output_tok_per_s", "Aggregate output tok/s (higher is better)", "{:.1f}"),
        ("ttft_p95_s", "TTFT p95 s (lower is better)", "{:.3f}"),
        ("per_stream_tok_per_s_p50", "Per-stream tok/s p50 (higher is better)", "{:.1f}"),
        ("failed", "Failed requests (must be 0)", "{:d}"),
    ]:
        lines += [f"## {title}", "",
                  "| Concurrency | " + " | ".join(labels) + " |",
                  "|---:|" + "---:|" * len(labels)]
        for c in sorted(common):
            cells = []
            for _, levels, _ in loaded:
                v = levels[c][metric]
                cells.append(fmt.format(v))
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        lines.append("")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    if mismatches:
        print("WARNING: settings differ:", mismatches)


if __name__ == "__main__":
    main()
