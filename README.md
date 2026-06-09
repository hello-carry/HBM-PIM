# HBM-PIM Evaluation Suite

This repository collects three independent
code bases used to evaluate processing-in-memory (PIM) on HBM memory stacks and
to compare it against conventional CPU and GPU processors across a common set of
data-parallel workloads (GEMM, GEMV, SpMV, 1-D convolution, histogram,
reduction, and vector addition).

## Components

| Directory | What it is | Build system | Details |
| --- | --- | --- | --- |
| [`HBM2_PIM/`](HBM2_PIM/README.md) | Cycle-accurate **HBM2**-PIM simulator derived from Samsung PIMSimulator + DRAMSim2. Functional and performance tests run as GoogleTest cases. | `scons` | [HBM2_PIM/README.md](HBM2_PIM/README.md) |
| [`HBM3_PIM/`](HBM3_PIM/README.md) | Cycle-accurate **HBM3**-PIM simulator built on Ramulator 2.0, driven by workload-specific trace generators. | `cmake` | [HBM3_PIM/README.md](HBM3_PIM/README.md) |
| [`GPU_CPU/`](GPU_CPU/README.md) | CPU and NVIDIA GPU reference implementations (the baselines the PIM simulators are compared against). Python (NumPy/CuPy) and C/CUDA/OpenMP. | `make` / `python` | [GPU_CPU/README.md](GPU_CPU/README.md) |

Each component is self-contained and has its own README with the exact build
commands, options, input data, and run examples. This page is only an overview —
**see the per-project README for anything specific.**

## Requirements At A Glance

- A C++ toolchain: C++17 for `HBM2_PIM`, C++20 (GCC 12+) for `HBM3_PIM`.
- `scons` + GoogleTest for `HBM2_PIM`; `cmake` 3.14+ for `HBM3_PIM`.
- Python 3 with NumPy/SciPy for the trace generators and Python baselines.
- An NVIDIA CUDA toolkit (`nvcc`) and a CUDA-capable GPU plus CuPy for the GPU
  baselines in `GPU_CPU/`.

The full, project-specific dependency lists are in the three READMEs.

## Quick Start

```bash
# HBM2-PIM simulator
cd HBM2_PIM && scons && ./sim --gtest_filter=PIMGEMMBenchFixture.*

# HBM3-PIM simulator
cmake -S HBM3_PIM -B HBM3_PIM/build && cmake --build HBM3_PIM/build -j
cp HBM3_PIM/build/ramulator2 HBM3_PIM/config_yaml/ramulator2
python HBM3_PIM/trace_gen/Macro_Benchmark/gemm_trace.py --rows 1024 --cols 1024 --n-cols 1024

# CPU / GPU baselines (see GPU_CPU/README.md for every workload)
python GPU_CPU/GEMM/gemm_cpu_gpu.py -M 1024 -K 1024 -N 1024
```

## Licensing

The simulators build on third-party projects (Samsung PIMSimulator, DRAMSim2,
Ramulator 2.0, AttAcc). Upstream license files are kept inside the respective
project directories — review them before redistributing.
