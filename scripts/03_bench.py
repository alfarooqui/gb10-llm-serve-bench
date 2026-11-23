#!/usr/bin/env python3
"""OpenAI-compatible serving benchmark: concurrency sweep with streaming metrics.

Standard library only (no pip install needed).

Writes to <out>/<profile>/<timestamp>/:
  raw.jsonl     one line per request
  summary.json  one object per concurrency level, plus meta
  summary.md    readable table
  env.json      host and server metadata
"""
import argparse
import datetime as dt
import http.client
import json
import pathlib
import platform
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Fixed prompt set. Do not change between runs or platforms, or results are not comparable.
PROMPTS = [
    "Explain the difference between tensor parallelism and pipeline parallelism for serving large language models.",
    "Write a short paragraph describing how a continuous batching scheduler admits requests into a running batch.",
    "List five practical things to check when a GPU inference server runs out of memory under load.",
    "Summarize why KV cache size grows with context length and batch size, in plain language.",
    "Describe the trade-offs between time-to-first-token and throughput when tuning an LLM server.",
    "Explain what NVFP4 quantization is and when it is worth using over FP8 or BF16.",
    "Give a step-by-step plan to benchmark an LLM endpoint fairly across two different GPUs.",
    "Describe how a reverse proxy should handle streaming responses from an LLM server.",
]
WARMUP_REQUESTS = 2


def pct(values, p):
    """Linear-interpolated percentile. Returns None for empty input."""
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def one_request(base_url, model, prompt, max_tokens, timeout_s):
    """Stream one chat completion and record TTFT, ITL, E2E and token count."""
    u = urllib.parse.urlsplit(base_url)
    conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    t0 = time.perf_counter()
    ttft = None
    last = None
    n_chunks = 0
    usage_tokens = None
    conn = conn_cls(u.hostname, u.port, timeout=timeout_s)
    try:
        conn.request("POST", "/v1/chat/completions", body=body, headers=headers)
        resp = conn.getresponse()
        if resp.status != 200:
            resp.read()
            return {"ok": False, "error": f"HTTP {resp.status}"}
        while True:
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage_tokens = obj["usage"].get("completion_tokens")
            choices = obj.get("choices") or []
            if choices and choices[0].get("delta", {}).get("content"):
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                last = now
                n_chunks += 1
    except Exception as e:  # noqa: BLE001 - record the failure and keep going
        return {"ok": False, "error": repr(e)}
    finally:
        conn.close()

    t1 = time.perf_counter()
    if ttft is None:
        return {"ok": False, "error": "no content tokens received"}
    out_tokens = usage_tokens or n_chunks
    itl = (last - (t0 + ttft)) / (n_chunks - 1) if n_chunks > 1 else None
    return {"ok": True, "ttft_s": ttft, "e2e_s": t1 - t0, "out_tokens": out_tokens, "itl_s": itl}


def run_level(base_url, model, concurrency, n_requests, max_tokens, timeout_s):
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n_requests)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(
            lambda p: one_request(base_url, model, p, max_tokens, timeout_s), prompts))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    ttfts = [r["ttft_s"] for r in ok]
    e2es = [r["e2e_s"] for r in ok]
    itls = [r["itl_s"] for r in ok if r["itl_s"] is not None]
    per_stream = [r["out_tokens"] / r["e2e_s"] for r in ok if r["e2e_s"] > 0]
    total_out = sum(r["out_tokens"] for r in ok)

    summary = {
        "concurrency": concurrency,
        "requests": n_requests,
        "failed": n_requests - len(ok),
        "wall_s": round(wall, 3),
        "req_per_s": round(len(ok) / wall, 3) if wall > 0 else 0,
        "aggregate_output_tok_per_s": round(total_out / wall, 2) if wall > 0 else 0,
        "per_stream_tok_per_s_p50": round(pct(per_stream, 50) or 0, 2),
        "ttft_p50_s": round(pct(ttfts, 50) or 0, 4),
        "ttft_p95_s": round(pct(ttfts, 95) or 0, 4),
        "itl_p50_ms": round((pct(itls, 50) or 0) * 1000, 2),
        "itl_p95_ms": round((pct(itls, 95) or 0) * 1000, 2),
        "e2e_p50_s": round(pct(e2es, 50) or 0, 4),
        "e2e_p95_s": round(pct(e2es, 95) or 0, 4),
    }
    raw = [dict(r, concurrency=concurrency) for r in results]
    return summary, raw


def host_info():
    info = {
        "hostname": platform.node(),
        "arch": platform.machine(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        info["gpu"] = q.stdout.strip().splitlines() if q.returncode == 0 else "nvidia-smi error"
    except Exception:  # noqa: BLE001
        info["gpu"] = "nvidia-smi unavailable"
    return info


def server_info(base_url):
    out = {}
    for path in ("/version", "/v1/models"):
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=10) as r:
                out[path] = json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            out[path] = repr(e)
    return out


def write_markdown(path, profile, model, summaries):
    lines = [
        f"# Benchmark: {profile}",
        "",
        f"Model: `{model}`",
        "",
        "| Conc | Req/s | Agg out tok/s | Per-stream tok/s (p50) | TTFT p50 s | TTFT p95 s | ITL p50 ms | ITL p95 ms | E2E p50 s | E2E p95 s | Failed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['concurrency']} | {s['req_per_s']} | {s['aggregate_output_tok_per_s']} | "
            f"{s['per_stream_tok_per_s_p50']} | {s['ttft_p50_s']} | {s['ttft_p95_s']} | "
            f"{s['itl_p50_ms']} | {s['itl_p95_ms']} | {s['e2e_p50_s']} | {s['e2e_p95_s']} | {s['failed']} |"
        )
    lines += [
        "",
        "Valid level: Failed = 0. Compare runs on the same model, prompts and max_tokens only.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--levels", default="1,2,4,8,16")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=600.0, help="per-request timeout, seconds")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = pathlib.Path(args.out) / args.profile / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Warmup: {WARMUP_REQUESTS} requests at concurrency 1 (discarded)")
    run_level(args.base_url, args.model, 1, WARMUP_REQUESTS, args.max_tokens, args.timeout)

    summaries, raw_all = [], []
    for c in levels:
        n = max(8, 4 * c)
        print(f"Concurrency {c}: {n} requests ...", flush=True)
        summary, raw = run_level(args.base_url, args.model, c, n, args.max_tokens, args.timeout)
        print(f"  agg {summary['aggregate_output_tok_per_s']} tok/s, "
              f"TTFT p50 {summary['ttft_p50_s']} s, failed {summary['failed']}")
        summaries.append(summary)
        raw_all.extend(raw)

    with open(out_dir / "raw.jsonl", "w") as f:
        for r in raw_all:
            f.write(json.dumps(r) + "\n")

    meta = {
        "profile": args.profile,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "prompts": len(PROMPTS),
        "levels": levels,
        "warmup_requests": WARMUP_REQUESTS,
    }
    (out_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "levels": summaries}, indent=2) + "\n")
    (out_dir / "env.json").write_text(
        json.dumps({"host": host_info(), "server": server_info(args.base_url)}, indent=2) + "\n")
    write_markdown(out_dir / "summary.md", args.profile, args.model, summaries)
    print(f"Wrote {out_dir}")
    total_failed = sum(s["failed"] for s in summaries)
    if total_failed:
        print(f"WARNING: {total_failed} requests failed. Results are not valid until this is 0.",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
