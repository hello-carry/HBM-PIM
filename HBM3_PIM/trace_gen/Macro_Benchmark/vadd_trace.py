import argparse
import math
import re
import subprocess
from pathlib import Path

# HBM3 configuration
n_channel = 16
n_pch = 2
n_rank = 2
n_bank = 4
n_bg = 4
n_row = pow(2, 14)
n_col = pow(2, 5)
prefetch_size = 32  # byte
n_mac = 16

# Granularity size
HBM_GS = {}
HBM_GS['col'] = prefetch_size
HBM_GS['row'] = n_col * HBM_GS['col']
HBM_GS['ba'] = n_row * HBM_GS['row']
HBM_GS['bg'] = n_bank * HBM_GS['ba']
HBM_GS['rank'] = n_bg * HBM_GS['bg']
HBM_GS['pch'] = n_rank * HBM_GS['rank']
HBM_GS['ch'] = n_pch * HBM_GS['pch']


def run_simulation(script_path, tck_ns, trace_filename):
    repo_root = Path(script_path).resolve().parent.parent.parent
    config_yaml_dir = repo_root / "config_yaml"
    ramulator2_bin = config_yaml_dir / "ramulator2"
    yaml_file = config_yaml_dir / "vadd_config.yaml"

    # Update trace path in yaml to point to the generated trace file
    yaml_text = yaml_file.read_text()
    yaml_text = re.sub(r'(?m)^(\s*path:\s*).*$',
                       f'\\1../trace_gen/trace/{trace_filename}', yaml_text, count=1)
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

    print(f"\n------   Vector Add Simulation Summary   ------")
    print(f"  memory_system_cycles : {cycles}")
    print(f"  tCK                  : {tck_ns} ns")
    print(f"  Total execution time : {time_us:.4f} us")
    print(f"-----------------------------------------------")


def generate_vadd_trace(vec_len, data_size, output_path):
    """
    AttAcc PIM vector add trace: C = A + B

    Flow per step (one 32B chunk across all banks in each channel):
      PIM_WR_GB A  -- load A chunk from DRAM into global buffer
      PIM_WR_GB B  -- load B chunk from DRAM into global buffer
      PIM_VADD     -- broadcast VADD to all banks in the channel

    Every 16 VADD steps (compute buffer full):
      PIM_MV_SB    -- flush compute buffer to softmax buffer (n_bg x n_rank x n_ch)
      PIM_WR_BK    -- write results back to DRAM bank (host-readable)

    Memory layout per channel (byte-addressed):
      A: [ch_base + 0               ... + col_per_bank * 32)
      B: [ch_base + col_per_bank*32 ... + 2*col_per_bank*32)
      C: [ch_base + 2*col_per_bank*32 ...]
    """

    cmd_wrgb_a = []
    cmd_wrgb_b = []
    cmd_vadd = []
    cmd_mvsb = []
    cmd_wrbk = []
    cmd_barrier = []
    ordered_cmds = []

    elems_per_32b = max(1, prefetch_size // data_size)
    # Number of 32B transactions per bank to cover the full vector
    col_per_bank = math.ceil(
        vec_len / (n_channel * n_pch * n_rank * n_bg * n_bank) / elems_per_32b
    )
    col_offset = col_per_bank      # B starts at col_per_bank * prefetch_size
    c_offset = 2 * col_per_bank   # C starts at 2 * col_per_bank * prefetch_size
    vadd_flush_threshold = 16

    # One BARRIER command per channel, used as a synchronization fence
    barrier = [f"PIM_BARRIER 0x{lch * HBM_GS['ch']:0>8x}" for lch in range(n_channel)]

    def emit_barrier():
        for cmd in barrier:
            cmd_barrier.append(cmd)
            ordered_cmds.append(cmd)

    def flush_and_writeback(flush_start):
        # BARRIER: ensure all VADDs are done before MV_SB
        # (no timing constraint between VADD and MVSB in the HBM3-PIM timing table)
        emit_barrier()

        # MV_SB: move compute results to softmax buffer
        for bg_idx in range(n_bg):
            for rank in range(n_rank):
                for lch in range(n_channel):
                    addr = lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + bg_idx * HBM_GS['bg']
                    cmd = f"PIM_MV_SB 0x{addr:0>8x}"
                    cmd_mvsb.append(cmd)
                    ordered_cmds.append(cmd)

        # WR_BK: write results back to DRAM (one per result slot per channel)
        flush_count = min(vadd_flush_threshold, col_per_bank - flush_start)
        for k in range(flush_count):
            for lch in range(n_channel):
                addr = lch * HBM_GS['ch'] + (c_offset + flush_start + k) * prefetch_size
                cmd = f"PIM_WR_BK 0x{addr:0>8x}"
                cmd_wrbk.append(cmd)
                ordered_cmds.append(cmd)

    vadd_step_count = 0
    flush_start = 0

    for col_idx in range(col_per_bank):
        for lch in range(n_channel):
            # WR_GB A: load A chunk (DRAM byte address)
            addr_a = lch * HBM_GS['ch'] + col_idx * prefetch_size
            cmd_a = f"PIM_WR_GB 0x{addr_a:0>8x}"
            cmd_wrgb_a.append(cmd_a)
            ordered_cmds.append(cmd_a)

        for lch in range(n_channel):
            # WR_GB B: load B chunk (DRAM byte address)
            addr_b = lch * HBM_GS['ch'] + (col_idx + col_offset) * prefetch_size
            cmd_b = f"PIM_WR_GB 0x{addr_b:0>8x}"
            cmd_wrgb_b.append(cmd_b)
            ordered_cmds.append(cmd_b)

        # BARRIER: ensure WR_GB A and B are done before VADD
        # (no timing constraint between WRGB and VADD in the HBM3-PIM timing table)
        emit_barrier()

        for lch in range(n_channel):
            # VADD: broadcast add to all banks in this channel
            vadd_addr = lch * HBM_GS['ch'] + col_idx * prefetch_size
            cmd_add = f"PIM_VADD 0x{vadd_addr:0>8x}"
            cmd_vadd.append(cmd_add)
            ordered_cmds.append(cmd_add)

        vadd_step_count += 1
        if vadd_step_count % vadd_flush_threshold == 0:
            flush_and_writeback(flush_start)
            flush_start += vadd_flush_threshold

    # Tail flush for remaining results
    if vadd_step_count % vadd_flush_threshold != 0:
        flush_and_writeback(flush_start)

    with open(output_path, 'w') as f:
        for cmd in ordered_cmds:
            f.write(cmd + "\n")

    print(f"Generated Vector Add trace: {output_path}")
    print(f"  col_per_bank : {col_per_bank}")
    print(f"  PIM_WR_GB (A): {len(cmd_wrgb_a)}")
    print(f"  PIM_WR_GB (B): {len(cmd_wrgb_b)}")
    print(f"  PIM_VADD     : {len(cmd_vadd)}")
    print(f"  PIM_MV_SB    : {len(cmd_mvsb)}")
    print(f"  PIM_WR_BK    : {len(cmd_wrbk)}")
    print(f"  PIM_BARRIER  : {len(cmd_barrier)}")
    print(f"  Total        : {len(ordered_cmds)} commands")


def main():
    parser = argparse.ArgumentParser(description="Generate vector add trace for AttAcc")

    parser.add_argument("-l", "--length", type=int, default=160000000,
                        help="Vector length (default: 160000000)")
    parser.add_argument("-db", "--dbyte", type=int, default=4,
                        help="Data type size in bytes, 4=FP32 (default: 4)")
    parser.add_argument("-o", "--output", type=str, default="",
                        help="Output trace file path (default: va_<length>.trace)")
    parser.add_argument("--run-sim", dest="run_sim", action="store_true", default=True,
                        help="Run simulator after generating trace (default: True)")
    parser.add_argument("--no-run-sim", dest="run_sim", action="store_false",
                        help="Skip simulator invocation")
    parser.add_argument("--tck-ns", type=float, default=0.769,
                        help="Clock period in ns (default: 0.769, HBM3_5.2Gbps)")

    args = parser.parse_args()

    if not args.output:
        args.output = f"va_{args.length}.trace"

    trace_name = Path(args.output).name
    trace_path = Path(__file__).resolve().parent.parent / "trace" / trace_name
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    print("------   Generate Vector Add Trace   ------")
    print(f"  Vector length: {args.length}")
    print(f"  Data size:     {args.dbyte} bytes")
    print(f"  Output:        {trace_path}")
    print("--------------------------------------------")

    generate_vadd_trace(args.length, args.dbyte, str(trace_path))

    if args.run_sim:
        run_simulation(__file__, args.tck_ns, trace_name)


if __name__ == "__main__":
    main()
