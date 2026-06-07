#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


# HBM3-PIM defaults aligned with this repository
DEFAULT_N_CHANNEL = 16
DEFAULT_N_PCH = 2
DEFAULT_N_RANK = 2
DEFAULT_N_BG = 4
DEFAULT_N_BANK = 4
DEFAULT_N_ROW = 2**14
DEFAULT_N_COL = 2**5
DEFAULT_PREFETCH_SIZE = 32  # bytes
DEFAULT_ELEMENT_BYTES = 4   # fp32 by default
DEFAULT_INPUT_LEN = 1_000_000
DEFAULT_KERNEL_LEN = 10


@dataclass(frozen=True)
class HBMConfig:
	n_channel: int = DEFAULT_N_CHANNEL
	n_pch: int = DEFAULT_N_PCH
	n_rank: int = DEFAULT_N_RANK
	n_bg: int = DEFAULT_N_BG
	n_bank: int = DEFAULT_N_BANK
	n_row: int = DEFAULT_N_ROW
	n_col: int = DEFAULT_N_COL
	prefetch_size: int = DEFAULT_PREFETCH_SIZE


def build_hbm_granularity(cfg: HBMConfig) -> dict[str, int]:
	gs: dict[str, int] = {}
	gs["col"] = cfg.prefetch_size
	gs["row"] = cfg.n_col * gs["col"]
	gs["ba"] = cfg.n_row * gs["row"]
	gs["bg"] = cfg.n_bank * gs["ba"]
	gs["rank"] = cfg.n_bg * gs["bg"]
	gs["pch"] = cfg.n_rank * gs["rank"]
	gs["ch"] = cfg.n_pch * gs["pch"]
	return gs


def format_cmd(op: str, addr: int) -> str:
	return f"{op} 0x{addr:0>8x}"


def conv1d_output_len(n: int, k: int, stride: int, padding: int, dilation: int) -> int:
	effective_k = dilation * (k - 1) + 1
	numer = n + 2 * padding - effective_k
	if numer < 0:
		return 0
	return numer // stride + 1


def validate_args(args: argparse.Namespace, cfg: HBMConfig) -> None:
	if args.n <= 0:
		raise ValueError("--n must be > 0")
	if args.k <= 0:
		raise ValueError("--k must be > 0")
	if args.stride <= 0:
		raise ValueError("--stride must be > 0")
	if args.dilation <= 0:
		raise ValueError("--dilation must be > 0")
	if args.padding < 0:
		raise ValueError("--padding must be >= 0")
	if args.element_bytes <= 0:
		raise ValueError("--element-bytes must be > 0")
	if cfg.prefetch_size <= 0:
		raise ValueError("--prefetch-size must be > 0")
	if cfg.prefetch_size % args.element_bytes != 0:
		raise ValueError("--prefetch-size must be divisible by --element-bytes")
	if args.active_channels <= 0 or args.active_channels > cfg.n_channel:
		raise ValueError("--active-channels must be in [1, n-channel]")
	if args.store_op not in {"PIM_WR_BK", "PIM_MV_SB", "ST", "NONE"}:
		raise ValueError("--store-op must be one of PIM_WR_BK/PIM_MV_SB/ST/NONE")
	if args.barrier_level not in {"none", "tap", "tile"}:
		raise ValueError("--barrier-level must be one of none/tap/tile")


def generate_conv1d_trace(args: argparse.Namespace, cfg: HBMConfig) -> list[str]:
	# Each PIM_MAC_AB triggers parallel computation across all banks within one channel, covering elems_per_mac_cmd output elements;
	gs = build_hbm_granularity(cfg)
	out_len = conv1d_output_len(args.n, args.k, args.stride, args.padding, args.dilation)
	if out_len <= 0:
		return []

	banks_per_channel = cfg.n_pch * cfg.n_rank * cfg.n_bg * cfg.n_bank
	elems_per_txn = cfg.prefetch_size // args.element_bytes
	elems_per_mac_cmd = elems_per_txn * banks_per_channel
	tiles = math.ceil(out_len / (elems_per_mac_cmd))

	lines: list[str] = []

	# Preload kernel weights into the global buffer so the later MAC loop can reuse them without repeated transfers.
	if args.preload_kernel:
		for tap in range(args.k):
			for ch in range(args.active_channels):
				addr = args.kernel_base_addr + ch * gs["ch"] + tap * gs["col"]
				lines.append(format_cmd("PIM_WR_GB", addr))
		if args.barrier_level != "none":
			for ch in range(args.active_channels):
				lines.append(format_cmd("PIM_BARRIER", args.input_base_addr + ch * gs["ch"]))

	for tile_idx in range(tiles):
		for tap in range(args.k):
			# Derive the input start element index from the output position (accounting for stride, dilation, padding).
			input_elem_index = tile_idx * elems_per_mac_cmd * args.stride + tap * args.dilation - args.padding
			# The left zero-padding region has no valid input address; clamp to 0.
			if input_elem_index < 0:
				input_elem_index = 0

			# Convert the element index to a cache-line (prefetch-granularity) index to get the row offset within HBM.
			line_idx = (input_elem_index * args.element_bytes) // cfg.prefetch_size

			for ch in range(args.active_channels):
				addr = args.input_base_addr + ch * gs["ch"] + line_idx * gs["col"]
				lines.append(format_cmd("PIM_MAC_AB", addr))

			if args.barrier_level == "tap":
				for ch in range(args.active_channels):
					lines.append(format_cmd("PIM_BARRIER", args.input_base_addr + ch * gs["ch"]))

		# Write back this tile's accumulated result. PIM_MV_SB issues one command per bank group; other ops issue one command per channel.
		if args.store_op != "NONE":
			if args.store_op in {"PIM_WR_BK", "ST"}:
				for ch in range(args.active_channels):
					addr = args.output_base_addr + ch * gs["ch"] + tile_idx * gs["col"]
					lines.append(format_cmd(args.store_op, addr))
			else:  # PIM_MV_SB
				for ch in range(args.active_channels):
					for pch in range(cfg.n_pch):
						for rank in range(cfg.n_rank):
							for bg in range(cfg.n_bg):
								addr = (
									args.output_base_addr
									+ ch * gs["ch"]
									+ pch * gs["pch"]
									+ rank * gs["rank"]
									+ bg * gs["bg"]
									+ tile_idx * gs["col"]
								)
								lines.append(format_cmd("PIM_MV_SB", addr))

		if args.barrier_level == "tile":
			for ch in range(args.active_channels):
				lines.append(format_cmd("PIM_BARRIER", args.output_base_addr + ch * gs["ch"]))

	return lines


def summarize(args: argparse.Namespace, cfg: HBMConfig, trace_lines: list[str]) -> None:
	out_len = conv1d_output_len(args.n, args.k, args.stride, args.padding, args.dilation)
	mac_cnt = sum(1 for c in trace_lines if c.startswith("PIM_MAC_AB"))
	wrgb_cnt = sum(1 for c in trace_lines if c.startswith("PIM_WR_GB"))
	barrier_cnt = sum(1 for c in trace_lines if c.startswith("PIM_BARRIER"))
	writeback_cnt = sum(
		1
		for c in trace_lines
		if c.startswith("PIM_WR_BK") or c.startswith("PIM_MV_SB") or c.startswith("ST")
	)

	banks_per_channel = cfg.n_pch * cfg.n_rank * cfg.n_bg * cfg.n_bank
	elems_per_txn = cfg.prefetch_size // args.element_bytes
	elems_per_mac_cmd = elems_per_txn * banks_per_channel
	tiles = math.ceil(out_len / (elems_per_mac_cmd * args.active_channels)) if out_len > 0 else 0

	print("------ Generate 1D CONV trace (HBM3-PIM) ------")
	print(f"Input length N: {args.n}")
	print(f"Kernel length K: {args.k}")
	print(f"Stride: {args.stride}, Padding: {args.padding}, Dilation: {args.dilation}")
	print(f"Output length: {out_len}")
	print(f"Active channels: {args.active_channels}")
	print(f"Banks per channel (pCH*Rank*BG*Bank): {banks_per_channel}")
	print(f"Elements per MAC command: {elems_per_mac_cmd}")
	print(f"Output tiles: {tiles}")
	print(f"PIM_WR_GB: {wrgb_cnt}")
	print(f"PIM_MAC_AB: {mac_cnt}")
	print(f"Writeback ({args.store_op}): {writeback_cnt}")
	print(f"PIM_BARRIER: {barrier_cnt}")
	print(f"Total commands: {len(trace_lines)}")


def run_simulation(trace_path: Path, tck_ns: float) -> None:
	repo_root = Path(__file__).resolve().parent.parent.parent
	config_yaml_dir = repo_root / "config_yaml"
	ramulator2_bin = config_yaml_dir / "ramulator2"
	yaml_file = config_yaml_dir / "conv_config.yaml"

	trace_abs = trace_path.resolve()
	trace_rel = os.path.relpath(trace_abs, config_yaml_dir)

	yaml_text = yaml_file.read_text()
	yaml_text = re.sub(
		r"(?m)^(\s*path:\s*).*$",
		f"\\1{trace_rel}",
		yaml_text,
		count=1,
	)
	yaml_file.write_text(yaml_text)

	print(f"\nRunning: {ramulator2_bin.name} -f {yaml_file.name}")
	result = subprocess.run(
		[str(ramulator2_bin), "-f", yaml_file.name],
		cwd=str(config_yaml_dir),
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
	)

	if result.returncode != 0:
		print(f"Simulator error:\n{result.stderr}")
		return

	m = re.search(r"memory_system_cycles:\s*(\d+)", result.stdout)
	if not m:
		print("Could not parse memory_system_cycles from simulator output.")
		print(result.stdout)
		return

	cycles = int(m.group(1))
	time_us = cycles * tck_ns * 1e-3

	print(f"\n------   1D Conv Simulation Summary   ------")
	print(f"  memory_system_cycles : {cycles}")
	print(f"  tCK                  : {tck_ns} ns")
	print(f"  Total execution time : {time_us:.4f} us")
	print(f"--------------------------------------------")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Generate HBM3-PIM trace for 1D convolution (timing-oriented)."
	)

	# 1D Conv parameters
	parser.add_argument("--n", type=int, default=DEFAULT_INPUT_LEN, help="Input length")
	parser.add_argument("--k", type=int, default=DEFAULT_KERNEL_LEN, help="Kernel length")
	parser.add_argument("--stride", type=int, default=2, help="Convolution stride")
	parser.add_argument("--padding", type=int, default=4, help="Zero padding size")
	parser.add_argument("--dilation", type=int, default=1, help="Convolution dilation")
	parser.add_argument("--element-bytes", type=int, default=DEFAULT_ELEMENT_BYTES, help="Bytes per element")

	# Trace command policy
	parser.add_argument(
		"--store-op",
		type=str,
		default="PIM_WR_BK",
		help="Output writeback op: PIM_WR_BK | PIM_MV_SB | ST | NONE",
	)
	parser.add_argument(
		"--barrier-level",
		type=str,
		default="tile",
		help="Insert barrier at: none | tap | tile",
	)
	parser.add_argument(
		"--preload-kernel",
		action="store_true",
		help="Emit PIM_WR_GB to preload kernel weights once before MAC loop",
	)

	# Address mapping
	parser.add_argument("--input-base-addr", type=lambda x: int(x, 0), default=0x0)
	parser.add_argument("--kernel-base-addr", type=lambda x: int(x, 0), default=0x20000000)
	parser.add_argument("--output-base-addr", type=lambda x: int(x, 0), default=0x40000000)
	parser.add_argument("--active-channels", type=int, default=DEFAULT_N_CHANNEL)

	# HBM geometry
	parser.add_argument("--n-channel", type=int, default=DEFAULT_N_CHANNEL)
	parser.add_argument("--n-pch", type=int, default=DEFAULT_N_PCH)
	parser.add_argument("--n-rank", type=int, default=DEFAULT_N_RANK)
	parser.add_argument("--n-bg", type=int, default=DEFAULT_N_BG)
	parser.add_argument("--n-bank", type=int, default=DEFAULT_N_BANK)
	parser.add_argument("--n-row", type=int, default=DEFAULT_N_ROW)
	parser.add_argument("--n-col", type=int, default=DEFAULT_N_COL)
	parser.add_argument("--prefetch-size", type=int, default=DEFAULT_PREFETCH_SIZE)

	parser.add_argument("-o", "--output", type=Path, default=Path(__file__).resolve().parent.parent / "trace" / "conv1d.trace", help="Output trace path")

	# Simulation control
	parser.add_argument("--run-sim", dest="run_sim", action="store_true", default=True,
						help="Run ramulator2 after generating trace (default: True)")
	parser.add_argument("--no-run-sim", dest="run_sim", action="store_false",
						help="Skip simulator invocation")
	parser.add_argument("--tck-ns", type=float, default=0.769,
						help="Clock period in ns (default: 0.769, HBM3 5.2 Gbps)")

	return parser.parse_args()


def main() -> None:
	args = parse_args()
	cfg = HBMConfig(
		n_channel=args.n_channel,
		n_pch=args.n_pch,
		n_rank=args.n_rank,
		n_bg=args.n_bg,
		n_bank=args.n_bank,
		n_row=args.n_row,
		n_col=args.n_col,
		prefetch_size=args.prefetch_size,
	)

	validate_args(args, cfg)
	trace_lines = generate_conv1d_trace(args, cfg)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text("\n".join(trace_lines) + ("\n" if trace_lines else ""), encoding="utf-8")

	summarize(args, cfg, trace_lines)
	print(f"Trace written to: {args.output}")

	if args.run_sim:
		run_simulation(args.output, args.tck_ns)


if __name__ == "__main__":
	main()
