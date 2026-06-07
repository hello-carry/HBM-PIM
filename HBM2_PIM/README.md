# HBM2-PIM Simulator

`HBM2_PIM` is a cycle-accurate HBM2 processing-in-memory simulator derived from
Samsung's PIMSimulator and DRAMSim2. It models an HBM2-compatible memory stack
with programmable PIM blocks and provides functional and performance tests for
common data-parallel workloads.

## Main Features

- HBM2 timing, organization, command scheduling, and address mapping.
- PIM blocks with command, general-purpose, and scalar register files.
- Microbenchmarks for sequential, random, strided, and compute throughput.
- Macrobenchmarks for GEMM, GEMV, SpMV, 1-D convolution, histogram, reduction,
  and vector addition.
- An emulator API for replaying memory traces produced by PIM kernels.

## Directory Layout

```text
HBM2_PIM/
|-- src/                    Simulator implementation
|   `-- tests/TestCase/     Macrobenchmarks and microbenchmarks
|-- tools/emulator_api/     PIM emulator integration API
|-- ini/                    HBM2 device configuration
|-- data/                   Functional-test data and generators
|-- dump/                   Optional simulator output directories
|-- lib/                    Header-only helper libraries
|-- Sconstruct              SCons build description
|-- system_hbm*.ini         Memory-system configurations
`-- LICENSE-*               Upstream license files
```

## Requirements

- A C++17 compiler
- SCons
- GoogleTest development libraries
- Python 3 with NumPy when regenerating functional-test data

On Ubuntu, the main build dependencies can be installed with:

```bash
sudo apt install build-essential scons libgtest-dev
```

## Build

Run the build from this directory:

```bash
cd HBM2_PIM
scons
```

Useful build options:

```bash
# Performance-only mode without backing data storage
scons NO_STORAGE=1

# Exclude the emulator API
scons NO_EMUL=1

# Remove generated build products
scons -c
```

The default build produces the `sim` test executable and DRAMSim libraries.

## Run

List the available GoogleTest cases:

```bash
./sim --gtest_list_tests
```

Run one benchmark or a benchmark family:

```bash
./sim --gtest_filter=PIMGEMMBenchFixture.*
./sim --gtest_filter=PIMBenchFixture.*
./sim --gtest_filter=PIMSpMVBenchFixture.*
./sim --gtest_filter=PIMConv1DBenchFixture.*
./sim --gtest_filter=PIMHistBenchFixture.*
./sim --gtest_filter=PIMReductionBenchFixture.*
./sim --gtest_filter=PIMVectorAddFixture.*
```

The microbenchmarks under `src/tests/TestCase/MicroBenchmark/` measure PIM
compute throughput and sequential, random, and strided access bandwidth.

## Configuration And Data

- `system_hbm.ini` selects the memory-system organization and address mapping.
- `ini/HBM2_samsung_2M_16B_x64.ini` contains HBM2 timing parameters.
- `system_hbm_1ch.ini` and `system_hbm_64ch.ini` provide channel-count variants.
- `data/*/gen_*.py` regenerates NumPy arrays used by functional tests.
- `dump/` is used for optional generated data and validation output.


## Upstream And Licensing

This directory is based on Samsung PIMSimulator and DRAMSim2. See
`LICENSE-PIMSimulator` and `LICENSE-DRAMSIM2` before using or redistributing the
code.
