#!/usr/bin/env python3
import argparse
import math
import re
import subprocess
from pathlib import Path

# HBM3-PIM architecture parameters
n_channel = 16
n_pch = 2
n_rank = 2
n_bg = 4
n_bank = 4
prefetch_size = 32
data_size = 4  # fp32

# HBM address space granularity
HBM_GS = {}
HBM_GS['col'] = prefetch_size
HBM_GS['row'] = 32 * HBM_GS['col']
HBM_GS['ba'] = 16384 * HBM_GS['row']
HBM_GS['bg'] = n_bank * HBM_GS['ba']
HBM_GS['rank'] = n_bg * HBM_GS['bg']
HBM_GS['pch'] = n_rank * HBM_GS['rank']
HBM_GS['ch'] = n_pch * HBM_GS['pch']


def generate_reduce_trace(n_elem, output_path=None):
    if output_path is None:
        output_path = Path(__file__).resolve().parent.parent / "trace" / "red.trace"
    else:
        output_path = Path(output_path)

    total_banks = n_channel * n_pch * n_rank * n_bg * n_bank
    elems_per_mac = prefetch_size // data_size
    col_per_bank = math.ceil(n_elem / total_banks / elems_per_mac)

    cmds = []

    # Phase 1: Single-pass — each bank independently accumulates its data chunk
    for col_idx in range(col_per_bank):
        for ch in range(n_channel):
            addr = ch * HBM_GS['ch'] + col_idx * HBM_GS['col']
            cmds.append(f"PIM_MAC_AB 0x{addr:0>8x}")

    # Phase 2: Barrier — wait for all banks to finish accumulation
    for ch in range(n_channel):
        cmds.append(f"PIM_BARRIER 0x{ch * HBM_GS['ch']:0>8x}")

    # Phase 3: Collect partial sums from all banks
    for ch in range(n_channel):
        for pch_i in range(n_pch):
            for rank in range(n_rank):
                for bg in range(n_bg):
                    addr = (ch * HBM_GS['ch'] + pch_i * HBM_GS['pch']
                            + rank * HBM_GS['rank'] + bg * HBM_GS['bg'])
                    cmds.append(f"PIM_MV_SB 0x{addr:0>8x}")

    # Phase 4: Final barrier
    for ch in range(n_channel):
        cmds.append(f"PIM_BARRIER 0x{ch * HBM_GS['ch']:0>8x}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(cmds) + "\n", encoding="utf-8")

    mac_cnt     = sum(1 for c in cmds if c.startswith("PIM_MAC_AB"))
    mvsb_cnt    = sum(1 for c in cmds if c.startswith("PIM_MV_SB"))
    barrier_cnt = sum(1 for c in cmds if c.startswith("PIM_BARRIER"))

    print(f"------   Generate Reduction Trace   ------")
    print(f"  Input elements : {n_elem}")
    print(f"  col_per_bank   : {col_per_bank}")
    print(f"  PIM_MAC_AB     : {mac_cnt}")
    print(f"  PIM_MV_SB      : {mvsb_cnt}")
    print(f"  PIM_BARRIER    : {barrier_cnt}")
    print(f"  Total          : {len(cmds)} commands")
    print(f"  Output         : {output_path}")
    print(f"------------------------------------------")

    return output_path


def run_simulation(output_path: Path, tck_ns: float) -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    config_yaml_dir = repo_root / "config_yaml"
    ramulator2_bin = config_yaml_dir / "ramulator2"
    yaml_file = config_yaml_dir / "red_config.yaml"

    yaml_text = yaml_file.read_text()
    yaml_text = re.sub(r'(?m)^(\s*path:\s*).*$',
                       f'\\1{output_path.resolve().as_posix()}', yaml_text, count=1)
    yaml_file.write_text(yaml_text)

    print(f"Running: {ramulator2_bin.name} -f {yaml_file.name}")
    result = subprocess.run(
        [str(ramulator2_bin), "-f", yaml_file.name],
        cwd=str(config_yaml_dir),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    if result.returncode != 0:
        print(f"Simulator error:\n{result.stderr}")
        return

    m = re.search(r'memory_system_cycles:\s*(\d+)', result.stdout)
    if not m:
        print("Could not parse memory_system_cycles from simulator output.")
        print(result.stdout)
        return

    cycles = int(m.group(1))
    time_us = cycles * tck_ns * 1e-3

    print(f"\n------   Reduction Simulation Summary   ------")
    print(f"  memory_system_cycles : {cycles}")
    print(f"  tCK                  : {tck_ns} ns")
    print(f"  Total execution time : {time_us:.4f} us")
    print(f"----------------------------------------------")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HBM3-PIM reduction trace and optionally run simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-elem", type=int, default=6_300_000,
                        help="Number of input elements to reduce.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output trace file path (default: trace/red.trace).")
    parser.add_argument("--run-sim", dest="run_sim", action="store_true", default=True,
                        help="Run ramulator2 simulator after generating trace.")
    parser.add_argument("--no-run-sim", dest="run_sim", action="store_false",
                        help="Skip simulator invocation.")
    parser.add_argument("--tck-ns", type=float, default=0.769,
                        help="Clock period in ns (default: 0.769, HBM3 5.2 Gbps).")
    args = parser.parse_args()

    output_path = generate_reduce_trace(args.n_elem, args.output)

    if args.run_sim:
        run_simulation(output_path, args.tck_ns)


if __name__ == "__main__":
    main()
