#!/usr/bin/env python3
"""Sequential (contiguous) aggregation-bandwidth trace for HBM3-PIM.

Mapping-agnostic: emits only a plain contiguous byte-address stream
    addr_i = base + i * 32B
The HBM hierarchy (channel/pch/bank/row/col) is decoded entirely by the
simulator's AddrMapper.  Default mapping = Scheme8 (override with --scheme).
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List

PREFETCH_SIZE = 32                       # bytes per PIM_MAC_AB transaction
BANK_BROADCAST_FACTOR = 2 * 2 * 4 * 4    # banks covered by one PIM_MAC_AB (n_pch*n_rank*n_bg*n_bank)
READ_OP = "PIM_MAC_AB"
DEFAULT_SCHEME = "Scheme8"

THIS = Path(__file__).resolve()
TRACE_GEN_DIR = THIS.parent.parent               # ramulator2/trace_gen (script lives in trace_gen/Micro_Benchmark)
TRACE_DIR = TRACE_GEN_DIR / "trace"              # all generated traces live here
TRACE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR = TRACE_GEN_DIR.parent / "config_yaml"
BASE_YAML = CONFIG_DIR / "seq_agg_bdw.yaml"
RUN_YAML = CONFIG_DIR / "seq_agg_bdw_run.yaml"   # generated per run; base yaml left untouched
RAMULATOR_BIN = CONFIG_DIR / "ramulator2"
CYCLES_RE = re.compile(r"memory_system_cycles:\s*(\d+)")


def gen_trace(n_cmd: int, base: int = 0) -> List[str]:
    """Pure contiguous walk: addr = base + i*32B.  No hierarchy awareness."""
    return [f"{READ_OP} 0x{base + i * PREFETCH_SIZE:08x}" for i in range(n_cmd)]


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


def main() -> int:
    ap = argparse.ArgumentParser(description="HBM3-PIM sequential bandwidth trace")
    ap.add_argument("--scheme", default=DEFAULT_SCHEME, help="address mapping scheme (default: Scheme8)")
    ap.add_argument("--n-cmd", type=int, default=3125, help="number of PIM_MAC_AB commands")
    ap.add_argument("--tck-ns", type=float, default=0.769)
    ap.add_argument("-o", "--output", default="seq_trace.trace", help="trace filename under trace_gen/")
    ap.add_argument("--no-run-sim", dest="run_sim", action="store_false")
    ap.set_defaults(run_sim=True)
    args = ap.parse_args()

    trace_path = TRACE_DIR / args.output
    trace_path.write_text("\n".join(gen_trace(args.n_cmd)) + "\n")
    print(f"Generated sequential trace: {trace_path}  ({args.n_cmd} commands, addr = i*{PREFETCH_SIZE}B)")
    if not args.run_sim:
        return 0

    impl = f"HBM3-PIM-{args.scheme}"
    write_run_yaml(impl, args.output)
    cycles = run_sim()

    total_bytes = args.n_cmd * PREFETCH_SIZE * BANK_BROADCAST_FACTOR
    time_ns = cycles * args.tck_ns
    bw = bandwidth_gbps(args.n_cmd, cycles, args.tck_ns)
    print("------   Sequential Bandwidth Summary   ------")
    print(f"  Mapping     : {impl}")
    print(f"  Simulator   : {RAMULATOR_BIN}")
    print(f"  Config yaml : {RUN_YAML}")
    print(f"  Trace       : {trace_path}")
    print(f"  Commands    : {args.n_cmd}")
    print(f"  Bytes       : {total_bytes} B ({total_bytes / 1024 / 1024:.4f} MB)")
    print(f"  Cycles      : {cycles}")
    print(f"  tCK         : {args.tck_ns} ns")
    print(f"  Access time : {time_ns:.2f} ns ({time_ns * 1e-3:.4f} us)")
    print(f"  Bandwidth   : {bw:.4f} GB/s")
    print("----------------------------------------------")
    return 0


if __name__ == "__main__":
    sys.exit(main())
