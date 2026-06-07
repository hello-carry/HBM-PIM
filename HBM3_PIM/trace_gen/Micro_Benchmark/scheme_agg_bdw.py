#!/usr/bin/env python3
"""Mapping-agnostic strided bandwidth sweep for HBM3-PIM.

Design principle (per user): the trace generator only emits a plain strided byte
address sequence

    addr_i = base + i * stride_bytes          (i = 0 .. n_cmd-1)

It knows NOTHING about the HBM hierarchy (channel/pch/bank/row/col).  How those
bits are interpreted is entirely the job of the simulator's address decoder
(the HBM3-PIM-SchemeN AddrMapper).  The SAME trace is fed to every scheme; only
`AddrMapper.impl` in the yaml changes between runs.  This is a genuine
"one address stream, different decoders" comparison.

For each scheme x stride we run ramulator2, read memory_system_cycles, and report
bandwidth = (n_cmd * 32 B * bank_broadcast_factor) / (cycles * tCK).
Results are written to a.csv in the same column layout as bwd_amp.csv.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

PREFETCH_SIZE = 32                       # bytes per PIM_MAC_AB transaction
BANK_BROADCAST_FACTOR = 2 * 2 * 4 * 4    # n_pch*n_rank*n_bg*n_bank covered by one PIM_MAC_AB
READ_OP = "PIM_MAC_AB"

SCHEMES_DEFAULT = ["Scheme2", "Scheme6", "Scheme8", "Scheme11"]
SCHEME_IMPL = lambda s: f"HBM3-PIM-{s}"

THIS = Path(__file__).resolve()
TRACE_GEN_DIR = THIS.parent.parent
TRACE_DIR = TRACE_GEN_DIR / "trace"              # all generated traces live here
TRACE_DIR.mkdir(parents=True, exist_ok=True)
RAMULATOR_ROOT = TRACE_GEN_DIR.parent
CONFIG_DIR = RAMULATOR_ROOT / "config_yaml"
PROJECT_ROOT = RAMULATOR_ROOT.parent
BASE_YAML = CONFIG_DIR / "str2_agg_bdw.yaml"
RAMULATOR_BIN = CONFIG_DIR / "ramulator2"

STRIDE_BYTES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
STRIDE_LABELS = ["str=32", "str=64", "str=128", "str=256", "str=512", "str=1K",
                 "str=2K", "str=4K", "str=8K", "str=16K", "str=32K", "str=64K", "str=128K"]

CYCLES_RE = re.compile(r"memory_system_cycles:\s*(\d+)")


def gen_trace(mode, n_cmd: int, base: int = 0) -> List[str]:
    """mode = stride_bytes (int): addr = base + i*stride_bytes.
       mode = 'rand'           : addr = base + random_line*PREFETCH_SIZE over the span.
    No hierarchy awareness whatsoever -- just byte addresses."""
    if mode == "rand":
        import random
        rng = random.Random(12345)
        span_lines = max(1, (n_cmd * STRIDE_BYTES[-1]) // PREFETCH_SIZE)  # same address span as the largest stride
        return [f"{READ_OP} 0x{base + rng.randrange(span_lines) * PREFETCH_SIZE:08x}" for _ in range(n_cmd)]
    return [f"{READ_OP} 0x{base + i * mode:08x}" for i in range(n_cmd)]


def patch_yaml(impl: str, trace_name: str) -> None:
    text = BASE_YAML.read_text()
    text, n1 = re.subn(r"(AddrMapper:\s*\n\s*impl:\s*)[\w\-]+",
                       lambda m: m.group(1) + impl, text, count=1)
    text, n2 = re.subn(r"(?m)^(\s*path:\s*)\.\./trace_gen/[\w./-]+\.trace\s*$",
                       lambda m: m.group(1) + f"../trace_gen/trace/{trace_name}", text, count=1)
    if n1 != 1 or n2 != 1:
        raise RuntimeError(f"yaml patch failed (impl={n1}, path={n2})")
    BASE_YAML.write_text(text)


def run_sim() -> int:
    res = subprocess.run([str(RAMULATOR_BIN), "-f", str(BASE_YAML)],
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", default=",".join(SCHEMES_DEFAULT))
    ap.add_argument("--n-cmd", type=int, default=3125,
                    help="commands per run, fixed across strides (=> fixed data, like HBM2)")
    ap.add_argument("--tck-ns", type=float, default=0.769)
    args = ap.parse_args()
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]

    # Generate the mapping-agnostic traces ONCE; reuse for every scheme.
    trace_files: Dict[str, str] = {}
    for mode, label in list(zip(STRIDE_BYTES, STRIDE_LABELS)) + [("rand", "rand")]:
        name = f"plain_{'rand' if mode == 'rand' else f'K{mode}B'}_N{args.n_cmd}.trace"
        (TRACE_DIR / name).write_text("\n".join(gen_trace(mode, args.n_cmd)) + "\n")
        trace_files[label] = name
    print(f"Generated {len(trace_files)} plain stride traces ({args.n_cmd} cmds each)")

    bak = BASE_YAML.with_suffix(".yaml.scheme_bak")
    shutil.copy(BASE_YAML, bak)
    results: Dict[str, Dict[str, float]] = {}
    try:
        for scheme in schemes:
            print(f"==== {scheme} ({SCHEME_IMPL(scheme)}) ====")
            row: Dict[str, float] = {}
            for label in STRIDE_LABELS + ["rand"]:
                patch_yaml(SCHEME_IMPL(scheme), trace_files[label])
                cycles = run_sim()
                bw = bandwidth_gbps(args.n_cmd, cycles, args.tck_ns)
                row[label] = bw
                print(f"  {label:>7s}: cycles={cycles:>8d}  BW={bw:10.4f} GB/s")
            row["seq"] = row["str=32"]          # seq == smallest stride
            results[scheme] = row
    finally:
        shutil.copy(bak, BASE_YAML)
        bak.unlink()
        print("restored str2_agg_bdw.yaml")

    cols = ["seq"] + STRIDE_LABELS + ["rand"]
    print("\n====== Aggregate Bandwidth (GB/s) ======")
    header = f"{'mapping policy':<16}" + "".join(f"{c:>11}" for c in cols)
    print(header)
    print("-" * len(header))
    for scheme in schemes:
        r = results[scheme]
        print(f"{scheme:<16}" + "".join(f"{r[c]:>11.4f}" for c in cols))
    print("=" * len(header))
    return 0


if __name__ == "__main__":
    sys.exit(main())
