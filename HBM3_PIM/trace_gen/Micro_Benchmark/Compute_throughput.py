#!/usr/bin/env python3
import argparse
import subprocess
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


# HBM3-PIM geometry. This follows gen_trace_attacc_bank.py.
N_CHANNEL = 16
N_PCH = 2
N_RANK = 2
N_BANK = 4
N_BG = 4
N_ROW = 1 << 14
N_COL = 1 << 5
PREFETCH_SIZE = 32  # byte
DEFAULT_TCK_NS = 1e6 / (5200 / 4) / 1000


@dataclass(frozen=True)
class DType:
    name: str
    element_bytes: int


@dataclass(frozen=True)
class Operation:
    name: str
    trace_cmd: str
    ops_per_element: int
    note: str


@dataclass(frozen=True)
class CaseStats:
    op: str
    dtype: str
    trace_cmd: str
    element_bytes: int
    elements_per_cmd: int
    ops_per_cmd: int
    compute_commands: int
    total_ops: int
    trace_path: Path
    yaml_path: Optional[Path] = None
    cycles: Optional[int] = None
    time_us: Optional[float] = None
    gops: Optional[float] = None


DTYPES: Dict[str, DType] = {
    "int8": DType("INT8", 1),
    "fp16": DType("FP16", 2),
    "fp32": DType("FP32", 4),
}

OPERATIONS: Dict[str, Operation] = {
    "add": Operation(
        "ADD",
        "PIM_VADD",
        1,
        "Vector add. One element addition is counted as 1 OP.",
    ),
    "mul": Operation(
        "MUL",
        "PIM_MAC_AB",
        1,
        "No standalone VMUL exists in this simulator; use MACAB datapath as multiply-only and count 1 OP per element.",
    ),
    "mac": Operation(
        "MAC",
        "PIM_MAC_AB",
        2,
        "Multiply-accumulate. One element MAC is counted as 2 OPs.",
    ),
}


def build_hbm_granularity() -> Dict[str, int]:
    gs: Dict[str, int] = {}
    gs["col"] = PREFETCH_SIZE
    gs["row"] = N_COL * gs["col"]
    gs["ba"] = N_ROW * gs["row"]
    gs["bg"] = N_BANK * gs["ba"]
    gs["rank"] = N_BG * gs["bg"]
    gs["pch"] = N_RANK * gs["rank"]
    gs["ch"] = N_PCH * gs["pch"]
    gs["hbm"] = N_CHANNEL * gs["ch"]
    return gs


HBM_GS = build_hbm_granularity()
BANKS_PER_CHANNEL = N_PCH * N_RANK * N_BG * N_BANK


def parse_list(values: Optional[List[str]], valid: Dict[str, object], default_all: Iterable[str]) -> List[str]:
    if values is None:
        return list(default_all)
    out: List[str] = []
    for value in values:
        key = value.lower()
        if key == "all":
            return list(default_all)
        if key not in valid:
            raise ValueError(f"Unknown value: {value}. Valid values: {', '.join(valid)}")
        out.append(key)
    return out


def yaml_quote(value: Path) -> str:
    text = str(value)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_config_yaml(
    yaml_path: Path,
    trace_path: Path,
    timing_preset: str,
    refresh_manager: str,
    max_addr: int,
) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = f"""Frontend:
  impl: PIMLoadStoreTrace
  path: {yaml_quote(trace_path.resolve())}
  clock_ratio: 1

  Translation:
    impl: NoTranslation
    max_addr: {max_addr}

MemorySystem:
  impl: PIMDRAM
  clock_ratio: 1
  DRAM:
    impl: HBM3-PIM
    org:
      preset: HBM3_8Gb_2R
      channel: {N_CHANNEL}
    timing:
      preset: {timing_preset}

  Controller:
    impl: HBM3-PIM
    Scheduler:
      impl: PIM
    RefreshManager:
      impl: {refresh_manager}

  AddrMapper:
    impl: HBM3-PIM
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")


def generate_trace(
    op: Operation,
    dtype: DType,
    output_path: Path,
    commands_per_channel: int,
    active_channels: int,
    row_span_cols: int,
) -> CaseStats:
    if dtype.element_bytes <= 0 or PREFETCH_SIZE % dtype.element_bytes != 0:
        raise ValueError("dtype element size must divide the 32B prefetch size")
    if commands_per_channel <= 0:
        raise ValueError("--commands-per-channel must be > 0")
    if active_channels <= 0 or active_channels > N_CHANNEL:
        raise ValueError(f"--active-channels must be in [1, {N_CHANNEL}]")
    if row_span_cols <= 0 or row_span_cols > N_COL:
        raise ValueError(f"--row-span-cols must be in [1, {N_COL}]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    compute_commands = 0

    with output_path.open("w", encoding="utf-8") as trace_file:
        for step in range(commands_per_channel):
            col = step % row_span_cols
            col_addr = col * HBM_GS["col"]
            for ch in range(active_channels):
                addr = ch * HBM_GS["ch"] + col_addr
                trace_file.write(f"{op.trace_cmd} 0x{addr:0>8x}\n")
                compute_commands += 1

    elements_per_cmd = BANKS_PER_CHANNEL * (PREFETCH_SIZE // dtype.element_bytes)
    ops_per_cmd = elements_per_cmd * op.ops_per_element
    total_ops = compute_commands * ops_per_cmd

    return CaseStats(
        op=op.name,
        dtype=dtype.name,
        trace_cmd=op.trace_cmd,
        element_bytes=dtype.element_bytes,
        elements_per_cmd=elements_per_cmd,
        ops_per_cmd=ops_per_cmd,
        compute_commands=compute_commands,
        total_ops=total_ops,
        trace_path=output_path,
    )


def run_simulation(ramulator_bin: Path, yaml_path: Path, cwd: Path) -> int:
    result = subprocess.run(
        [str(ramulator_bin), "-f", str(yaml_path)],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "ramulator2 failed\n"
            f"command: {ramulator_bin} -f {yaml_path}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    match = re.search(r"memory_system_cycles:\s*(\d+)", result.stdout)
    if not match:
        raise RuntimeError(
            "Could not parse memory_system_cycles from ramulator2 output\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return int(match.group(1))


def with_sim_result(stats: CaseStats, yaml_path: Path, cycles: int, tck_ns: float) -> CaseStats:
    time_us = cycles * tck_ns * 1e-3
    gops = stats.total_ops / (cycles * tck_ns)
    return CaseStats(
        op=stats.op,
        dtype=stats.dtype,
        trace_cmd=stats.trace_cmd,
        element_bytes=stats.element_bytes,
        elements_per_cmd=stats.elements_per_cmd,
        ops_per_cmd=stats.ops_per_cmd,
        compute_commands=stats.compute_commands,
        total_ops=stats.total_ops,
        trace_path=stats.trace_path,
        yaml_path=yaml_path,
        cycles=cycles,
        time_us=time_us,
        gops=gops,
    )


def print_case_generation(stats: CaseStats) -> None:
    print(f"Generated {stats.op}-{stats.dtype} trace: {stats.trace_path}")
    print(f"  command          : {stats.trace_cmd}")
    print(f"  element bytes    : {stats.element_bytes}")
    print(f"  elements/command : {stats.elements_per_cmd}")
    print(f"  OPs/command      : {stats.ops_per_cmd}")
    print(f"  compute commands : {stats.compute_commands}")
    print(f"  total OPs        : {stats.total_ops}")


def print_summary(results: List[CaseStats], tck_ns: float) -> None:
    print("\n------ HBM3-PIM Basic Operation Throughput ------")
    print(f"  tCK: {tck_ns:.6f} ns")
    header = (
        f"{'OP':<5} {'DType':<6} {'Cmd':<11} {'Cmds':>10} "
        f"{'Total OPs':>16} {'Cycles':>12} {'Time(us)':>12} {'GOPS':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        cycles = "-" if r.cycles is None else str(r.cycles)
        time_us = "-" if r.time_us is None else f"{r.time_us:.4f}"
        gops = "-" if r.gops is None else f"{r.gops:.3f}"
        print(
            f"{r.op:<5} {r.dtype:<6} {r.trace_cmd:<11} {r.compute_commands:>10} "
            f"{r.total_ops:>16} {cycles:>12} {time_us:>12} {gops:>12}"
        )
    print("-------------------------------------------------")
    print("OP counting: ADD/MUL = 1 OP per element, MAC = 2 OPs per element.")
    print("MUL uses PIM_MAC_AB because this simulator has no standalone PIM_VMUL request.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HBM3-PIM ADD/MUL/MAC traces and measure GOPS with ramulator2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        default=None,
        help="Operations to run: add mul mac all.",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=None,
        help="Data types to run: int8 fp16 fp32 all.",
    )
    parser.add_argument(
        "--commands-per-channel",
        type=int,
        default=4096,
        help="Compute commands emitted per active channel.",
    )
    parser.add_argument(
        "--active-channels",
        type=int,
        default=N_CHANNEL,
        help="Number of HBM channels used in the trace.",
    )
    parser.add_argument(
        "--row-span-cols",
        type=int,
        default=N_COL,
        help="Number of columns reused inside one open row for compute commands.",
    )
    parser.add_argument(
        "--run-sim",
        dest="run_sim",
        action="store_true",
        default=True,
        help="Run ramulator2 after generating each trace.",
    )
    parser.add_argument(
        "--no-run-sim",
        dest="run_sim",
        action="store_false",
        help="Only generate traces.",
    )
    parser.add_argument(
        "--ramulator-bin",
        type=Path,
        default=None,
        help="Path to ramulator2 binary. Default is ramulator2/config_yaml/ramulator2.",
    )
    parser.add_argument(
        "--timing-preset",
        default="HBM3_5.2Gbps",
        help="HBM3-PIM timing preset used in generated yaml.",
    )
    parser.add_argument(
        "--refresh-manager",
        choices=["No", "AllBankHBM3"],
        default="No",
        help="Refresh manager for the generated yaml. Use No for peak compute throughput.",
    )
    parser.add_argument(
        "--tck-ns",
        type=float,
        default=DEFAULT_TCK_NS,
        help="Clock period used to convert cycles to time.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    op_keys = parse_list(args.ops, OPERATIONS, OPERATIONS.keys())
    dtype_keys = parse_list(args.dtypes, DTYPES, DTYPES.keys())

    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent.parent
    config_yaml_dir = repo_root / "config_yaml"
    ramulator_bin = args.ramulator_bin or (config_yaml_dir / "ramulator2")
    trace_dir = Path(__file__).resolve().parent.parent / "trace"   # all generated traces live here
    trace_dir.mkdir(parents=True, exist_ok=True)

    if args.run_sim and not ramulator_bin.exists():
        raise FileNotFoundError(f"ramulator2 binary not found: {ramulator_bin}")

    print("------ Generate HBM3-PIM Basic Operation Traces ------")
    print(f"  ops/channel      : {args.commands_per_channel} compute commands")
    print(f"  active channels  : {args.active_channels}")
    print(f"  timing preset    : {args.timing_preset}")
    print(f"  refresh manager  : {args.refresh_manager}")
    print("------------------------------------------------------")

    results: List[CaseStats] = []
    max_addr = HBM_GS["hbm"]

    for op_key in op_keys:
        op = OPERATIONS[op_key]
        for dtype_key in dtype_keys:
            dtype = DTYPES[dtype_key]
            trace_path = trace_dir / f"hbm3_pim_{op_key}_{dtype_key}.trace"
            yaml_path = config_yaml_dir / f"hbm3_pim_{op_key}_{dtype_key}.yaml"

            stats = generate_trace(
                op=op,
                dtype=dtype,
                output_path=trace_path,
                commands_per_channel=args.commands_per_channel,
                active_channels=args.active_channels,
                row_span_cols=args.row_span_cols,
            )
            print_case_generation(stats)

            if args.run_sim:
                write_config_yaml(
                    yaml_path=yaml_path,
                    trace_path=trace_path,
                    timing_preset=args.timing_preset,
                    refresh_manager=args.refresh_manager,
                    max_addr=max_addr,
                )
                print(f"Running: {ramulator_bin} -f {yaml_path}")
                cycles = run_simulation(ramulator_bin, yaml_path.resolve(), config_yaml_dir)
                stats = with_sim_result(stats, yaml_path, cycles, args.tck_ns)
                print(f"  memory_system_cycles : {cycles}")
                print(f"  total execution time : {stats.time_us:.4f} us")
                print(f"  throughput           : {stats.gops:.3f} GOPS")

            results.append(stats)

    print_summary(results, args.tck_ns)


if __name__ == "__main__":
    main()
