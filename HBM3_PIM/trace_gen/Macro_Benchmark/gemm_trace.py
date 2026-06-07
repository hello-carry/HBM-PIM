import argparse
import math
import os
import re
import subprocess
from pathlib import Path

# HBM3-PIM architecture parameters
n_channel = 16
n_pch = 2
n_rank = 2
n_bg = 4
n_bank = 4
n_mac = 16
prefetch_size = 32
data_size = 4

# HBM address space granularity
HBM_GS = {}
HBM_GS['col'] = prefetch_size
HBM_GS['row'] = 32 * HBM_GS['col']
HBM_GS['ba'] = 16384 * HBM_GS['row']
HBM_GS['bg'] = n_bank * HBM_GS['ba']
HBM_GS['rank'] = n_bg * HBM_GS['bg']
HBM_GS['pch'] = n_rank * HBM_GS['rank']
HBM_GS['ch'] = n_pch * HBM_GS['pch']


def generate_gemm_trace(m, k, n, output_path=None, active_channels=None):
    if active_channels is None:
        active_channels = n_channel
    if active_channels <= 0 or active_channels > n_channel:
        raise ValueError(f"active_channels must be in [1, {n_channel}], got {active_channels}")

    if output_path is None:
        output_path = Path(__file__).resolve().parent.parent / "trace" / f"gemm_{m}x{k}x{n}.trace"
    else:
        output_path = Path(output_path)

    barrier = [f"PIM_BARRIER 0x{lch * HBM_GS['ch']:0>8x}" for lch in range(active_channels)]
    total_cmd = []

    k_per_out = math.ceil(k / (n_bank * n_mac))
    out_chunks = math.ceil(m / (n_pch * n_rank * n_bg))
    wrgb_cols = math.ceil(k / n_bank / n_mac)
    n_per_channel = math.ceil(n / active_channels)

    wrgb_count = 0
    mac_count = 0
    mvsb_count = 0

    for n_local in range(n_per_channel):
        # ===== Phase 1: WR_GB — each channel loads its assigned B columns into the GEMV buffer =====
        for ba_idx in range(n_bank):
            for col_idx in range(wrgb_cols):
                for lch in range(active_channels):
                    if lch * n_per_channel + n_local >= n:
                        continue
                    addr = lch * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + col_idx
                    total_cmd.append(f"PIM_WR_GB 0x{addr:0>8x}")
                    wrgb_count += 1
        total_cmd.extend(barrier)

        # ===== Phase 2: interleave MAC_AB + MV_SB, flushing every 16 output chunks =====
        for out_idx in range(out_chunks):
            for in_idx in range(k_per_out):
                combined_idx = in_idx + out_idx * k_per_out
                for lch in range(active_channels):
                    if lch * n_per_channel + n_local >= n:
                        continue
                    addr = lch * HBM_GS['ch'] + combined_idx * HBM_GS['col']
                    total_cmd.append(f"PIM_MAC_AB 0x{addr:0>8x}")
                    mac_count += 1

            if out_idx % 16 == 15 or out_idx == out_chunks - 1:
                total_cmd.extend(barrier)
                for bg_idx in range(n_bg):
                    for rank in range(n_rank):
                        for lch in range(active_channels):
                            if lch * n_per_channel + n_local >= n:
                                continue
                            bank_addr = lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + bg_idx * HBM_GS['bg']
                            total_cmd.append(f"PIM_MV_SB 0x{bank_addr:0>8x}")
                            mvsb_count += 1
                if not (n_local == n_per_channel - 1 and out_idx == out_chunks - 1):
                    total_cmd.extend(barrier)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(total_cmd) + "\n", encoding="utf-8")

    barrier_cnt = len(total_cmd) - wrgb_count - mac_count - mvsb_count

    print(f"------   Generate GEMM Trace   ------")
    print(f"  Matrix           : {m} x {k} x {n}")
    print(f"  active_channels  : {active_channels} / {n_channel}")
    print(f"  n_per_channel    : {n_per_channel}")
    print(f"  out_chunks       : {out_chunks}")
    print(f"  k_per_out        : {k_per_out}")
    print(f"  PIM_WR_GB        : {wrgb_count}")
    print(f"  PIM_MAC_AB       : {mac_count}")
    print(f"  PIM_MV_SB        : {mvsb_count}")
    print(f"  PIM_BARRIER      : {barrier_cnt}")
    print(f"  Total            : {len(total_cmd)} commands")
    print(f"  Output           : {output_path}")
    print(f"-------------------------------------")

    return output_path


def run_simulation(output_path: Path, tck_ns: float) -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    config_yaml_dir = repo_root / "config_yaml"
    ramulator2_bin = config_yaml_dir / "ramulator2"
    yaml_file = config_yaml_dir / "gemm_config.yaml"

    rel_path = os.path.relpath(str(output_path.resolve()), str(config_yaml_dir.resolve()))
    yaml_text = yaml_file.read_text()
    yaml_text = re.sub(r'(?m)^(\s*path:\s*).*$', f'\\1{rel_path}', yaml_text, count=1)
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

    print(f"\n------   GEMM Simulation Summary   ------")
    print(f"  memory_system_cycles : {cycles}")
    print(f"  tCK                  : {tck_ns} ns")
    print(f"  Total execution time : {time_us:.4f} us")
    print(f"-----------------------------------------")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GEMM PIM trace for HBM3-PIM and optionally run simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", "--rows", type=int, default=1024, help="Output dimension M.")
    parser.add_argument("-k", "--cols", type=int, default=1024, help="Input dimension K.")
    parser.add_argument("-n", "--n-cols", type=int, default=1024, help="Output dimension N.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output trace file path (default: trace_gen/gemm_<M>x<K>x<N>.trace).")
    parser.add_argument("--run-sim", dest="run_sim", action="store_true", default=True,
                        help="Run ramulator2 simulator after generating trace.")
    parser.add_argument("--no-run-sim", dest="run_sim", action="store_false",
                        help="Skip simulator invocation.")
    parser.add_argument("--tck-ns", type=float, default=0.769,
                        help="Clock period in ns (default: 0.769, HBM3 5.2 Gbps).")
    parser.add_argument("--n-channel", dest="n_channel", type=int, default=n_channel,
                        help=f"Active channels in trace (1..{n_channel}). The yaml channel field stays unchanged.")
    args = parser.parse_args()

    output_path = generate_gemm_trace(args.rows, args.cols, args.n_cols, args.output,
                                      active_channels=args.n_channel)

    if args.run_sim:
        run_simulation(output_path, args.tck_ns)


if __name__ == "__main__":
    main()
