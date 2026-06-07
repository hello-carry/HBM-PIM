#!/usr/bin/env python3
"""Strided aggregation-bandwidth trace + sweep for HBM3-PIM.

Mapping-agnostic: emits only a plain strided byte-address stream
    addr_i = base + i * stride_bytes
The HBM hierarchy (channel/pch/bank/row/col) is decoded entirely by the
simulator's AddrMapper.  Default mapping = Scheme8 (override with --scheme).

Default runs a stride sweep (32B .. 128KB) and prints one bandwidth per stride;
use --stride-bytes for a single stride, and -o to also dump the sweep to CSV.
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import List

PREFETCH_SIZE = 32                       # bytes per PIM_MAC_AB transaction
BANK_BROADCAST_FACTOR = 2 * 2 * 4 * 4    # banks covered by one PIM_MAC_AB (n_pch*n_rank*n_bg*n_bank)
READ_OP = "PIM_MAC_AB"
DEFAULT_SCHEME = "Scheme8"

STRIDE_BYTES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
STRIDE_LABELS = ["str=32", "str=64", "str=128", "str=256", "str=512", "str=1K",
                 "str=2K", "str=4K", "str=8K", "str=16K", "str=32K", "str=64K", "str=128K"]

THIS = Path(__file__).resolve()
TRACE_GEN_DIR = THIS.parent.parent               # ramulator2/trace_gen (script lives in trace_gen/Micro_Benchmark)
TRACE_DIR = TRACE_GEN_DIR / "trace"              # all generated traces live here
TRACE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR = TRACE_GEN_DIR.parent / "config_yaml"
BASE_YAML = CONFIG_DIR / "str2_agg_bdw.yaml"
RUN_YAML = CONFIG_DIR / "str2_agg_bdw_run.yaml"   # generated per run; base yaml left untouched
RAMULATOR_BIN = CONFIG_DIR / "ramulator2"
CYCLES_RE = re.compile(r"memory_system_cycles:\s*(\d+)")


def gen_trace(stride_bytes: int, n_cmd: int, base: int = 0) -> List[str]:
    """Pure strided walk: addr = base + i*stride_bytes.  No hierarchy awareness."""
    return [f"{READ_OP} 0x{base + i * stride_bytes:08x}" for i in range(n_cmd)]


def write_run_yaml(impl: str, trace_name: str) -> None:
    """Derive RUN_YAML from BASE_YAML with the chosen scheme + trace path; base untouched."""
    text = BASE_YAML.read_text()
    text, n1 = re.subn(r"(AddrMapper:\s*\n\s*impl:\s*)[\w\-]+",
                       lambda m: m.group(1) + impl, text, count=1)
    text, n2 = re.subn(r"(?m)^(\s*path:\s*)\.\./trace_gen/[\w./-]+\.trace\s*$",
                       lambda m: m.group(1) + f"../trace_gen/trace/{trace_name}", text, count=1)
    if n1 != 1 or n2 != 1:
        raise RuntimeError(f"yaml patch failed (impl={n1}, path={n2})")
    RUN_YAML.write_text(text)


def run_sim() -> int:
    res = subprocess.run([str(RAMULATOR_BIN), "-f", str(RUN_YAML)],
                         cwd=str(CONFIG_DIR), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        print(res.stdout)
        raise RuntimeError(f"ramulator2 exited {res.returncode}")
    m = CYCLES_RE.search(res.stdout)
    if not m:
        print(res.stdout)
        raise RuntimeError("memory_system_cycles not found")
    return int(m.group(1))


def bandwidth_gbps(n_cmd: int, cycles: int, tck_ns: float) -> float:
    total_bytes = n_cmd * PREFETCH_SIZE * BANK_BROADCAST_FACTOR
    return (total_bytes / (cycles * tck_ns * 1e-9)) / 1e9


def run_one(stride_bytes: int, impl: str, n_cmd: int, tck_ns: float):
    """Returns (bandwidth_gbps, cycles, trace_path)."""
    trace_name = f"str_trace_K{stride_bytes}B.trace"
    trace_path = TRACE_DIR / trace_name
    trace_path.write_text("\n".join(gen_trace(stride_bytes, n_cmd)) + "\n")
    write_run_yaml(impl, trace_name)
    cycles = run_sim()
    return bandwidth_gbps(n_cmd, cycles, tck_ns), cycles, trace_path


def main() -> int:
    ap = argparse.ArgumentParser(description="HBM3-PIM strided bandwidth trace + sweep")
    ap.add_argument("--scheme", default=DEFAULT_SCHEME, help="address mapping scheme (default: Scheme8)")
    ap.add_argument("--stride-bytes", type=int, default=0, help="single stride in bytes (0 => full sweep)")
    ap.add_argument("--n-cmd", type=int, default=3125, help="commands per run (fixed across strides)")
    ap.add_argument("--tck-ns", type=float, default=0.769)
    ap.add_argument("-o", "--output", default="", help="optional CSV path for the sweep")
    ap.add_argument("--no-run-sim", dest="run_sim", action="store_false")
    ap.set_defaults(run_sim=True)
    args = ap.parse_args()

    impl = f"HBM3-PIM-{args.scheme}"

    # single-stride mode
    if args.stride_bytes > 0:
        trace_name = f"str_trace_K{args.stride_bytes}B.trace"
        (TRACE_DIR / trace_name).write_text("\n".join(gen_trace(args.stride_bytes, args.n_cmd)) + "\n")
        print(f"Generated strided trace: {TRACE_DIR / trace_name}  (stride={args.stride_bytes}B, {args.n_cmd} cmds)")
        if not args.run_sim:
            return 0
        bw, cycles, tpath = run_one(args.stride_bytes, impl, args.n_cmd, args.tck_ns)
        total_bytes = args.n_cmd * PREFETCH_SIZE * BANK_BROADCAST_FACTOR
        time_ns = cycles * args.tck_ns
        print("------   Strided Bandwidth Summary   ------")
        print(f"  Mapping     : {impl}")
        print(f"  Simulator   : {RAMULATOR_BIN}")
        print(f"  Config yaml : {RUN_YAML}")
        print(f"  Trace       : {tpath}")
        print(f"  Stride      : {args.stride_bytes} B")
        print(f"  Commands    : {args.n_cmd}")
        print(f"  Bytes       : {total_bytes} B ({total_bytes / 1024 / 1024:.4f} MB)")
        print(f"  Cycles      : {cycles}")
        print(f"  tCK         : {args.tck_ns} ns")
        print(f"  Access time : {time_ns:.2f} ns ({time_ns * 1e-3:.4f} us)")
        print(f"  Bandwidth   : {bw:.4f} GB/s")
        print("-------------------------------------------")
        return 0

    # sweep mode
    if not args.run_sim:
        for sb in STRIDE_BYTES:
            name = f"str_trace_K{sb}B.trace"
            (TRACE_DIR / name).write_text("\n".join(gen_trace(sb, args.n_cmd)) + "\n")
        print(f"Generated {len(STRIDE_BYTES)} strided traces (no sim).")
        return 0

    bws = {}
    total_bytes = args.n_cmd * PREFETCH_SIZE * BANK_BROADCAST_FACTOR
    print(f"==== {impl} stride sweep ====")
    print(f"  Simulator   : {RAMULATOR_BIN}")
    print(f"  Config yaml : {RUN_YAML}")
    print(f"  Commands    : {args.n_cmd}   Bytes/run: {total_bytes} B ({total_bytes / 1024 / 1024:.4f} MB)   tCK: {args.tck_ns} ns")
    for sb, lab in zip(STRIDE_BYTES, STRIDE_LABELS):
        bw, cycles, _ = run_one(sb, impl, args.n_cmd, args.tck_ns)
        bws[lab] = bw
        time_ns = cycles * args.tck_ns
        print(f"  {lab:>7s}: cycles={cycles:>8d}  time={time_ns * 1e-3:9.4f} us  BW={bw:10.4f} GB/s")

    if args.output:
        cols = ["mapping policy", "seq"] + STRIDE_LABELS
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerow([args.scheme, f"{bws['str=32']:.4f}"] + [f"{bws[l]:.4f}" for l in STRIDE_LABELS])
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
