#!/usr/bin/env python3
import argparse
import math
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse

# HBM3-PIM architecture parameters
n_channel = 16
n_pch = 2
n_rank = 2
n_bg = 4
n_bank = 4
n_row = 1 << 14
n_col = 1 << 5
prefetch_size = 32
data_size = 4

N_BANK_PER_CH = n_pch * n_rank * n_bg * n_bank   # 64
TOTAL_BANKS   = n_channel * N_BANK_PER_CH         # 1024
BURST_ELEMS   = prefetch_size // data_size        # 8 fp32 / burst
GB_ENTRIES    = 16                                # GEMV buffer capacity of one channel
TILE_ELEMS    = GB_ENTRIES * BURST_ELEMS          # one column tile = 128 x elements
DEFAULT_MAC_BATCH = GB_ENTRIES - 1                # reserve 1 slot for the input x / kernel, 15 output accumulation slots
DEFAULT_INDEX_BASE_ADDR = 4 << 20                 # per-bank local offset: 4 MiB
DEFAULT_Y_BASE_ADDR = 8 << 20                     # per-bank local offset: 8 MiB

HBM_GS = {}
HBM_GS['col']  = prefetch_size
HBM_GS['row']  = n_col   * HBM_GS['col']
HBM_GS['ba']   = n_row   * HBM_GS['row']
HBM_GS['bg']   = n_bank  * HBM_GS['ba']
HBM_GS['rank'] = n_bg    * HBM_GS['bg']
HBM_GS['pch']  = n_rank  * HBM_GS['rank']
HBM_GS['ch']   = n_pch   * HBM_GS['pch']


def bank_base_addr(bank_id):
    ch   = bank_id % n_channel
    rest = bank_id // n_channel
    ba   = rest % n_bank
    rest //= n_bank
    bg   = rest % n_bg
    rest //= n_bg
    rank = rest % n_rank
    pch  = rest // n_rank
    return (ch * HBM_GS['ch'] + pch * HBM_GS['pch']
            + rank * HBM_GS['rank'] + bg * HBM_GS['bg'] + ba * HBM_GS['ba'])


def bank_channel(bank_id):
    return bank_id % n_channel


class TraceEmitter:
    def __init__(self, output_path, max_commands=0):
        self.output_path = Path(output_path)
        self.max_commands = max_commands
        self.counts = defaultdict(int)
        self.total = 0
        self._fh = None

    def __enter__(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fh is not None:
            self._fh.close()
        return False

    def emit(self, op, addr):
        if self.max_commands and self.total >= self.max_commands:
            raise RuntimeError(f"Trace command limit reached: {self.max_commands}")
        self._fh.write(f"{op} 0x{addr:0>8x}\n")
        self.counts[op] += 1
        self.total += 1


def emit_global_barrier(emitter):
    for ch in range(n_channel):
        emitter.emit("PIM_BARRIER", ch * HBM_GS['ch'])


def emit_channel_barrier(emitter, ch):
    emitter.emit("PIM_BARRIER", ch * HBM_GS['ch'])


def emit_mvsb_all(emitter):
    for ch in range(n_channel):
        for pch_i in range(n_pch):
            for rank_i in range(n_rank):
                for bg_i in range(n_bg):
                    addr = (ch * HBM_GS['ch'] + pch_i * HBM_GS['pch']
                            + rank_i * HBM_GS['rank'] + bg_i * HBM_GS['bg'])
                    emitter.emit("PIM_MV_SB", addr)


def emit_mvsb_bank(emitter, bank_id):
    emitter.emit("PIM_MV_SB", bank_base_addr(bank_id))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an HBM3-PIM SpMV timing trace (row-partition + column-tile + MACPB).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("matrix", nargs='?', type=Path, default=None,
                        help="Input sparse matrix (.npz or .mtx). Optional with --rows/--cols/--density.")
    parser.add_argument("--rows", type=int, default=28924)
    parser.add_argument("--cols", type=int, default=28924)
    parser.add_argument("--density", type=float, default=0.0123)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--x-base-addr", type=lambda s: int(s, 0), default=0,
                        help="Byte offset of x within bank0; one copy of x is stored in bank0 of each channel")
    parser.add_argument("--value-base-addr", type=lambda s: int(s, 0), default=0,
                        help="CSR value array per-bank local byte offset")
    parser.add_argument("--index-base-addr", type=lambda s: int(s, 0), default=DEFAULT_INDEX_BASE_ADDR,
                        help="CSR column-index array per-bank local byte offset")
    parser.add_argument("--y-base-addr", type=lambda s: int(s, 0), default=DEFAULT_Y_BASE_ADDR,
                        help="Output y array per-bank local byte offset")
    parser.add_argument("--model", choices=["csr", "tiled-compressed"], default="csr",
                        help="Trace model: csr is row-aware; tiled-compressed reproduces the old optimistic model")
    parser.add_argument("--x-cache-policy", choices=["none", "tile"], default="none",
                        help="CSR model x loading: none gathers x for every nnz; tile reuses one x burst per row/tile/slot")
    parser.add_argument("--lane-pack", type=int, default=1,
                        help="CSR value MAC packing. 1 emits one MAC per nnz; >1 is an optimistic lane pack")
    parser.add_argument("--mac-batch", type=int, default=DEFAULT_MAC_BATCH,
                        help="CSR MACs accumulated before flushing a row partial sum")
    parser.add_argument("--y-write-op", choices=["PIM_WR_BK", "ST"], default="PIM_WR_BK",
                        help="Command used to write final y[row]")
    parser.add_argument("--index-read-op", choices=["PIM_WR_GB", "LD"], default="PIM_WR_GB",
                        help="Command used to model CSR column-index reads. PIM_WR_GB keeps the trace in the PIM pipeline; LD is available for experiments")
    parser.add_argument("--max-commands", type=int, default=0,
                        help="Optional safety limit for generated commands; 0 disables the limit")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--run-sim", dest="run_sim", action="store_true", default=True)
    parser.add_argument("--no-run-sim", dest="run_sim", action="store_false")
    parser.add_argument("--tck-ns", type=float, default=0.769)
    args = parser.parse_args()
    if args.lane_pack < 1:
        raise ValueError("--lane-pack must be >= 1")
    if args.mac_batch < 1:
        raise ValueError("--mac-batch must be >= 1")
    for name in ("value_base_addr", "index_base_addr", "y_base_addr", "x_base_addr"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    return args


def load_sparse_matrix(matrix_path):
    suffix = matrix_path.suffix.lower()
    if suffix == ".npz":
        matrix = scipy.sparse.load_npz(matrix_path)
    elif suffix == ".mtx":
        matrix = scipy.io.mmread(matrix_path)
    else:
        raise ValueError(f"Unsupported format: {matrix_path.suffix}. Use .npz or .mtx.")
    matrix = matrix.tocsr()
    matrix.sort_indices()
    return matrix


def generate_random_matrix(rows, cols, density, seed=None):
    rng = np.random.default_rng(seed)
    matrix = scipy.sparse.random(rows, cols, density=density, format="csr",
                                  random_state=seed, data_rvs=rng.standard_normal)
    matrix.eliminate_zeros()
    return matrix


def assign_rows_to_banks(matrix):
    nrows = matrix.shape[0]
    if matrix.nnz == 0 or nrows == 0:
        return np.zeros(nrows, dtype=np.int64)
    row_nnz = np.diff(matrix.indptr).astype(np.int64)
    prev = np.concatenate([[0], np.cumsum(row_nnz)[:-1]])
    target = matrix.nnz / TOTAL_BANKS
    return np.minimum(np.floor(prev / target).astype(np.int64), TOTAL_BANKS - 1)


def compute_workload(matrix, bank_of_row):
    ncols = matrix.shape[1]
    num_tiles = max(math.ceil(ncols / TILE_ELEMS), 1)

    coo = matrix.tocoo()
    if coo.nnz == 0:
        bursts = np.zeros((TOTAL_BANKS, num_tiles), dtype=np.int64)
        used_x = np.zeros((n_channel, num_tiles, GB_ENTRIES), dtype=bool)
        return bursts, used_x, num_tiles

    bnk  = bank_of_row[coo.row]
    tile = coo.col // TILE_ELEMS
    slot = (coo.col % TILE_ELEMS) // BURST_ELEMS
    chs  = bnk % n_channel

    nnz_grid = np.zeros((TOTAL_BANKS, num_tiles), dtype=np.int64)
    np.add.at(nnz_grid, (bnk, tile), 1)
    bursts = (nnz_grid + BURST_ELEMS - 1) // BURST_ELEMS

    used_x = np.zeros((n_channel, num_tiles, GB_ENTRIES), dtype=bool)
    used_x[chs, tile, slot] = True

    return bursts, used_x, num_tiles


def estimate_tiled_compressed_macs(matrix, bank_of_row):
    bursts, _, _ = compute_workload(matrix, bank_of_row)
    return int(bursts.sum())


def generate_tiled_compressed_trace(matrix, x_base_addr, emitter):
    bank_of_row = assign_rows_to_banks(matrix)
    bursts, used_x, num_tiles = compute_workload(matrix, bank_of_row)

    mac_cursor = np.zeros(TOTAL_BANKS, dtype=np.int64)

    for tile in range(num_tiles):
        # Phase A: each channel loads the x bursts used by this tile into the GB
        for ch in range(n_channel):
            for slot in range(GB_ENTRIES):
                if not used_x[ch, tile, slot]:
                    continue
                burst_idx = tile * GB_ENTRIES + slot
                addr = (x_base_addr
                        + ch * HBM_GS['ch']
                        + (burst_idx // n_col) * HBM_GS['row']
                        + (burst_idx % n_col) * HBM_GS['col'])
                emitter.emit("PIM_WR_GB", addr)

        emit_global_barrier(emitter)

        # Phase B: number of bursts per bank in this tile
        tile_macs = 0
        for bank_id in range(TOTAL_BANKS):
            n_b = int(bursts[bank_id, tile])
            if n_b == 0:
                continue
            base = bank_base_addr(bank_id)
            for _ in range(n_b):
                s = int(mac_cursor[bank_id])
                addr = base + (s // n_col) * HBM_GS['row'] + (s % n_col) * HBM_GS['col']
                emitter.emit("PIM_MAC_PB", addr)
                mac_cursor[bank_id] += 1
                tile_macs += 1

        if tile_macs == 0:
            continue

        emit_global_barrier(emitter)

        # Phase C: once the GB is full, move the partial sums to the softmax buffer
        emit_mvsb_all(emitter)

        emit_global_barrier(emitter)

    return {
        "num_tiles": num_tiles,
        "compressed_macs": int(emitter.counts["PIM_MAC_PB"]),
        "semantic_macs": matrix.nnz,
    }


def generate_csr_trace(matrix, args, emitter):
    """CSR row-aware model with explicit col-index reads, x gathers, MACs, row flushes, and y writes."""
    matrix = matrix.tocsr()
    matrix.sort_indices()
    bank_of_row = assign_rows_to_banks(matrix)
    compressed_macs = estimate_tiled_compressed_macs(matrix, bank_of_row)

    indptr = matrix.indptr
    indices = matrix.indices
    value_cursor = np.zeros(TOTAL_BANKS, dtype=np.int64)

    stats = defaultdict(int)
    stats["num_tiles"] = max(math.ceil(matrix.shape[1] / TILE_ELEMS), 1)
    stats["compressed_macs"] = compressed_macs

    for row in range(matrix.shape[0]):
        row_begin = int(indptr[row])
        row_end = int(indptr[row + 1])
        row_nnz = row_end - row_begin
        if row_nnz == 0:
            continue

        bank_id = int(bank_of_row[row])
        ch = bank_channel(bank_id)
        bank_base = bank_base_addr(bank_id)
        row_tiles = set()
        row_tile_slots = set()
        loaded_x_slots = set()
        macs_in_batch = 0

        stats["active_rows"] += 1

        for chunk_begin in range(row_begin, row_end, args.lane_pack):
            chunk_end = min(chunk_begin + args.lane_pack, row_end)
            value_addr = None

            for pos in range(chunk_begin, chunk_end):
                local_value_idx = int(value_cursor[bank_id])
                value_cursor[bank_id] += 1

                col = int(indices[pos])
                tile = col // TILE_ELEMS
                slot = (col % TILE_ELEMS) // BURST_ELEMS
                row_tiles.add(tile)
                row_tile_slots.add((tile, slot))

                if value_addr is None:
                    value_addr = bank_base + args.value_base_addr + local_value_idx * data_size

                index_addr = bank_base + args.index_base_addr + local_value_idx * 4
                emitter.emit(args.index_read_op, index_addr)
                stats["csr_index_reads"] += 1

                x_key = (tile, slot)
                if args.x_cache_policy == "none" or x_key not in loaded_x_slots:
                    x_burst = col // BURST_ELEMS
                    x_addr = args.x_base_addr + ch * HBM_GS['ch'] + x_burst * prefetch_size
                    emitter.emit("PIM_WR_GB", x_addr)
                    stats["x_gathers"] += 1
                    loaded_x_slots.add(x_key)

            emitter.emit("PIM_MAC_PB", value_addr)
            stats["matrix_macs"] += 1
            macs_in_batch += 1

            is_last_chunk = chunk_end >= row_end
            if macs_in_batch == args.mac_batch or is_last_chunk:
                emit_channel_barrier(emitter, ch)
                emit_mvsb_bank(emitter, bank_id)
                stats["partial_flushes"] += 1
                if is_last_chunk:
                    y_addr = bank_base + args.y_base_addr + row * data_size
                    emitter.emit(args.y_write_op, y_addr)
                    stats["y_writebacks"] += 1
                emit_channel_barrier(emitter, ch)
                macs_in_batch = 0

        stats["active_row_tiles"] += len(row_tiles)
        stats["active_row_tile_slots"] += len(row_tile_slots)

    stats["semantic_macs"] = stats["matrix_macs"]
    return stats


def run_simulation(output_path, tck_ns):
    repo_root = Path(__file__).resolve().parent.parent.parent
    config_yaml_dir = repo_root / "config_yaml"
    ramulator2_bin = config_yaml_dir / "ramulator2"
    yaml_file = config_yaml_dir / "spmv_config.yaml"

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
    print(f"\n------   SpMV Simulation Summary   ------")
    print(f"  memory_system_cycles : {cycles}")
    print(f"  tCK                  : {tck_ns} ns")
    print(f"  Total execution time : {time_us:.4f} us")
    print(f"-----------------------------------------")


def main():
    args = parse_args()
    output_path = args.output.resolve() if args.output else Path(__file__).resolve().parent.parent / "trace" / "spmv.trace"

    if args.matrix:
        matrix = load_sparse_matrix(args.matrix.resolve())
    else:
        matrix = generate_random_matrix(args.rows, args.cols, args.density, args.seed)

    with TraceEmitter(output_path, args.max_commands) as emitter:
        if args.model == "tiled-compressed":
            stats = generate_tiled_compressed_trace(matrix, args.x_base_addr, emitter)
        else:
            stats = generate_csr_trace(matrix, args, emitter)
        counts = emitter.counts
        total_commands = emitter.total

    print("------ Generate SpMV trace ------")
    print(f"  model       : {args.model}")
    print(f"  matrix      : {matrix.shape[0]} x {matrix.shape[1]}")
    print(f"  nnz         : {matrix.nnz}")
    print(f"  column tiles: {stats['num_tiles']}  (tile size = {TILE_ELEMS})")
    if args.model == "csr":
        print(f"  active rows : {stats['active_rows']}")
        print(f"  row tiles   : {stats['active_row_tiles']}")
        print(f"  row tile/slots: {stats['active_row_tile_slots']}")
        print(f"  CSR index reads: {stats['csr_index_reads']}")
        print(f"  x gathers   : {stats['x_gathers']}")
        print(f"  partial flushes: {stats['partial_flushes']}")
        print(f"  y writebacks: {stats['y_writebacks']}")
        print(f"  lane pack   : {args.lane_pack}")
        print(f"  MAC batch   : {args.mac_batch}")
        print(f"  x cache     : {args.x_cache_policy}")
        print(f"  index op    : {args.index_read_op}")
    print(f"  compressed MAC est: {stats['compressed_macs']}")
    print(f"  semantic MACs     : {stats['semantic_macs']}")
    print(f"  LD          : {counts['LD']}")
    print(f"  PIM_WR_GB   : {counts['PIM_WR_GB']}")
    print(f"  PIM_MAC_PB  : {counts['PIM_MAC_PB']}")
    print(f"  PIM_MV_SB   : {counts['PIM_MV_SB']}")
    print(f"  PIM_WR_BK   : {counts['PIM_WR_BK']}")
    print(f"  ST          : {counts['ST']}")
    print(f"  PIM_BARRIER : {counts['PIM_BARRIER']}")
    print(f"  Total       : {total_commands} commands")
    print(f"  Output      : {output_path}")

    if args.run_sim:
        run_simulation(output_path, args.tck_ns)


if __name__ == "__main__":
    main()
