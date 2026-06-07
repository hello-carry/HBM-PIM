#!/usr/bin/env python3
import argparse
import math
import random
import re
import subprocess
from pathlib import Path

# HBM3-PIM architecture parameters
n_channel     = 16
n_pch         = 2
n_rank        = 2
n_bg          = 4
n_bank        = 4
prefetch_size = 32  # bytes

# HBM address space granularity
HBM_GS = {}
HBM_GS['col']  = prefetch_size
HBM_GS['row']  = 32    * HBM_GS['col']
HBM_GS['ba']   = 16384 * HBM_GS['row']
HBM_GS['bg']   = n_bank  * HBM_GS['ba']
HBM_GS['rank'] = n_bg    * HBM_GS['bg']
HBM_GS['pch']  = n_rank  * HBM_GS['rank']
HBM_GS['ch']   = n_pch   * HBM_GS['pch']

N_BANK_PER_CH = n_pch * n_rank * n_bg * n_bank  # 64
MAC_BATCH     = 16                              # GEMV output buffer depth


def bank_offset(bank_local):
    bpr  = n_bg * n_bank
    bppc = n_rank * bpr
    pch  = bank_local // bppc;  rem = bank_local % bppc
    rank = rem // bpr;           rem = rem % bpr
    bg   = rem // n_bank;        ba  = rem % n_bank
    return pch * HBM_GS['pch'] + rank * HBM_GS['rank'] + bg * HBM_GS['bg'] + ba * HBM_GS['ba']


def emit_mvsb(cmds):
    for pch_i in range(n_pch):
        for rank_i in range(n_rank):
            for bg_idx in range(n_bg):
                for lch in range(n_channel):
                    addr = (lch * HBM_GS['ch'] + pch_i * HBM_GS['pch']
                            + rank_i * HBM_GS['rank'] + bg_idx * HBM_GS['bg'])
                    cmds.append(f"PIM_MV_SB 0x{addr:0>8x}")


def generate_hist_trace(n, bins, depth, seed, element_bytes, counter_bytes,
                        input_base, local_hist_base, global_hist_base, output_path):
    elems_per_txn = prefetch_size // element_bytes
    total_banks   = n_channel * N_BANK_PER_CH
    col_per_bank  = math.ceil(n / total_banks / elems_per_txn)
    max_val       = (1 << depth) - 1

    # Generate random input + round-robin assignment (element_map is used for the RMW in Phase 2)
    rng = random.Random(seed)
    element_map = []
    for i in range(n):
        v    = rng.randint(0, max_val)
        slot = i % total_banks
        ch   = slot % n_channel
        bl   = slot // n_channel
        element_map.append((ch, bl, v))

    cmds    = []
    barrier = [f"PIM_BARRIER 0x{ch * HBM_GS['ch']:0>8x}" for ch in range(n_channel)]

    # ---- Phase 1: scan — one batch of MAC_AB per 16 cols → flush GEMV
    for col_idx in range(col_per_bank):
        for ch in range(n_channel):
            addr = input_base + ch * HBM_GS['ch'] + col_idx * HBM_GS['col']
            cmds.append(f"PIM_MAC_AB 0x{addr:0>8x}")

        if (col_idx + 1) % MAC_BATCH == 0 or col_idx == col_per_bank - 1:
            cmds += barrier
            emit_mvsb(cmds)
            cmds += barrier

    # ---- Phase 2: local hist RMW — one batch per 16 elements
    for batch_start in range(0, n, MAC_BATCH):
        batch = element_map[batch_start : batch_start + MAC_BATCH]
        addrs = [
            local_hist_base + ch * HBM_GS['ch'] + bank_offset(bl)
            + ((v * bins) >> depth) * counter_bytes
            for ch, bl, v in batch
        ]

        for a in addrs:
            cmds.append(f"PIM_MAC_AB 0x{a:0>8x}")
        cmds += barrier
        emit_mvsb(cmds)
        cmds += barrier
        for a in addrs:
            cmds.append(f"PIM_WR_BK 0x{a:0>8x}")
        cmds += barrier

    # ---- Phase 3: merge — 16 MAC_AB per bin (exactly fills the GEMV) → flush → write global
    for bin_idx in range(bins):
        for ch in range(n_channel):
            addr = local_hist_base + ch * HBM_GS['ch'] + bin_idx * counter_bytes
            cmds.append(f"PIM_MAC_AB 0x{addr:0>8x}")
        cmds += barrier
        emit_mvsb(cmds)
        cmds += barrier
        cmds.append(f"PIM_WR_BK 0x{global_hist_base + bin_idx * counter_bytes:0>8x}")
    cmds += barrier

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for cmd in cmds:
            f.write(cmd + "\n")

    # golden verification
    golden = [0] * bins
    for _, _, v in element_map:
        golden[(v * bins) >> depth] += 1

    mac_cnt  = sum(1 for c in cmds if c.startswith("PIM_MAC_AB"))
    wrbk_cnt = sum(1 for c in cmds if c.startswith("PIM_WR_BK"))
    mvsb_cnt = sum(1 for c in cmds if c.startswith("PIM_MV_SB"))
    bar_cnt  = sum(1 for c in cmds if c.startswith("PIM_BARRIER"))

    print(f"------   Generate HIST Trace   ------")
    print(f"  N             : {n}")
    print(f"  bins / depth  : {bins} / {depth}  (values in [0, {max_val}])")
    print(f"  col_per_bank  : {col_per_bank}")
    print(f"  PIM_MAC_AB    : {mac_cnt}")
    print(f"  PIM_WR_BK     : {wrbk_cnt}")
    print(f"  PIM_MV_SB     : {mvsb_cnt}")
    print(f"  PIM_BARRIER   : {bar_cnt}")
    print(f"  Total         : {len(cmds)} commands")
    print(f"  Hist sum      : {sum(golden)} (expected {n})")
    print(f"  Output        : {output_path}")
    print(f"-------------------------------------")


def run_simulation(output_path: Path, tck_ns: float) -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    config_yaml_dir = repo_root / "config_yaml"
    ramulator2_bin = config_yaml_dir / "ramulator2"
    yaml_file = config_yaml_dir / "hist_config.yaml"

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

    print(f"\n------   Histogram Simulation Summary   ------")
    print(f"  memory_system_cycles : {cycles}")
    print(f"  tCK                  : {tck_ns} ns")
    print(f"  Total execution time : {time_us:.4f} us")
    print(f"----------------------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Generate HBM3-PIM histogram trace")
    parser.add_argument("--n",                 type=int,              default=10_000_000,    help="Input array length (default: 1000000)")
    parser.add_argument("--bins",              type=int,              default=256,          help="Histogram bins (default: 256)")
    parser.add_argument("--depth",             type=int,              default=12,           help="Input bit depth, values in [0, 2^depth-1] (default: 12)")
    parser.add_argument("--seed",              type=int,              default=0)
    parser.add_argument("--element-bytes",     type=int,              default=4)
    parser.add_argument("--counter-bytes",     type=int,              default=4)
    parser.add_argument("--input-base-addr",   type=lambda x: int(x, 0), default=0x00000000)
    parser.add_argument("--local-hist-base-addr",  type=lambda x: int(x, 0), default=0x20000000)
    parser.add_argument("--global-hist-base-addr", type=lambda x: int(x, 0), default=0x40000000)
    parser.add_argument("-o", "--output",      default=str(Path(__file__).resolve().parent.parent / "trace" / "hist.trace"))
    parser.add_argument("--run-sim", dest="run_sim", action="store_true", default=True,
                        help="Run ramulator2 simulator after generating trace.")
    parser.add_argument("--no-run-sim", dest="run_sim", action="store_false",
                        help="Skip simulator invocation.")
    parser.add_argument("--tck-ns", type=float, default=0.769,
                        help="Clock period in ns (default: 0.769, HBM3 5.2 Gbps).")
    args = parser.parse_args()

    generate_hist_trace(
        n=args.n, bins=args.bins, depth=args.depth, seed=args.seed,
        element_bytes=args.element_bytes, counter_bytes=args.counter_bytes,
        input_base=args.input_base_addr, local_hist_base=args.local_hist_base_addr,
        global_hist_base=args.global_hist_base_addr, output_path=args.output,
    )

    if args.run_sim:
        run_simulation(Path(args.output).resolve(), args.tck_ns)


if __name__ == "__main__":
    main()
