# HBM3-PIM Simulator

`HBM3_PIM` is a cycle-accurate processing-in-memory simulator built on
Ramulator 2.0. It adds an HBM3-PIM device model, PIM-aware address mapping,
controller and scheduling logic, command tracing, and workload-specific trace
generators.

## Main Features

- HBM3 timing and organization modeled through the Ramulator 2.0 framework.
- All-bank, bank-group, and per-bank PIM MAC commands.
- PIM data movement, vector addition, reduction, synchronization, and model
  configuration commands.
- Power-constrained and unconstrained HBM3-PIM timing presets.
- Macrobenchmark trace generators for GEMM, GEMV, SpMV, 1-D convolution,
  histogram, reduction, and vector addition.
- Microbenchmarks for compute throughput and sequential, random, and strided
  aggregation bandwidth.

## Directory Layout

```text
HBM3_PIM/
|-- src/                     Simulator implementation
|-- config_yaml/             Simulator configurations
|-- trace_gen/
|   |-- Macro_Benchmark/     Workload trace generators
|   `-- Micro_Benchmark/     Throughput and bandwidth trace generators
|-- ext/                     Vendored third-party source dependencies
|-- patches/                 Patches documenting Ramulator changes
|-- perf_comparison/         Cross-simulator comparison utilities
|-- resources/               Integration helpers
`-- CMakeLists.txt
```

## Requirements

- A C++20 compiler such as GCC 12
- CMake 3.14 
- Python 3.8 
- NumPy and SciPy for the Python trace generators

## Build

From the repository root:

```bash
cmake -S HBM3_PIM -B HBM3_PIM/build
cmake --build HBM3_PIM/build -j
cp HBM3_PIM/build/ramulator2 HBM3_PIM/config_yaml/ramulator2
```

The build also produces `HBM3_PIM/libramulator.so`. The Python generators look
for the simulator executable at `config_yaml/ramulator2`.

## Generate Traces And Run Simulations

The generators run the simulator by default. Pass `--no-run-sim` to generate a
trace without simulating it.

```bash
# GEMM
python HBM3_PIM/trace_gen/Macro_Benchmark/gemm_trace.py \
  --rows 1024 --cols 1024 --n-cols 1024

# GEMV
python HBM3_PIM/trace_gen/Macro_Benchmark/gemv_trace.py \
  --rows 8192 --cols 1024

```

Run the simulator directly when a matching trace already exists:

```bash
cd HBM3_PIM/config_yaml
./ramulator2 -f spmv_config.yaml
```

The simulator reports `memory_system_cycles`. The trace generators use this
value and the configured HBM3 clock period to report latency, bandwidth, or
throughput.

## Microbenchmarks

```bash
python HBM3_PIM/trace_gen/Micro_Benchmark/Compute_throughput.py --help
python HBM3_PIM/trace_gen/Micro_Benchmark/seq_trace.py --help
python HBM3_PIM/trace_gen/Micro_Benchmark/ran_trace.py --help
python HBM3_PIM/trace_gen/Micro_Benchmark/str_trace.py --help
python HBM3_PIM/trace_gen/Micro_Benchmark/scheme_agg_bdw.py --help
```

Timing preset `HBM3_5.2Gbps` enables the modeled DRAM power constraint.
`HBM3_5.2Gbps_NPC` removes that constraint for peak-throughput experiments.
The preset and channel count are selected in `config_yaml/*.yaml`.

## Upstream

The simulator is based on Ramulator 2.0 and uses the HBM-PIM modeling approach
from AttAcc. Third-party source dependencies and their license files are kept
under `ext/`.
