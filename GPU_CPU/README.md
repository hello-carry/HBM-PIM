# CPU And GPU Baselines

`GPU_CPU` contains CPU and NVIDIA GPU reference implementations used to compare
the HBM-PIM simulators with conventional processors. The benchmarks cover dense,
sparse, and data-parallel workloads.

## Workloads

| Directory | Workload |
| --- | --- |
| `CONV/` | 1-D convolution |
| `GEMM/` | Matrix-matrix multiplication |
| `GEMV/` | Matrix-vector multiplication |
| `HIST/` | Histogram |
| `SRED/` | Scalar reduction |
| `SpMV/` | Sparse matrix-vector multiplication |
| `VA/` | Vector addition |

The C/CUDA workloads store their conventional implementations under
`baselines/cpu/` and `baselines/gpu/`. Shared command-line parsing, timing, and
data helpers are under each workload's `support/` directory.

## Python Benchmarks

Requirements:

- Python 3
- NumPy
- CuPy built for the installed CUDA version
- An NVIDIA CUDA-capable GPU

Examples:

```bash
python GPU_CPU/CONV/conv_cpu_gpu.py \
  --input-size 1000000 --kernel-size 10 --stride 2 --padding 4

python GPU_CPU/GEMM/gemm_cpu_gpu.py \
  -M 1024 -K 1024 -N 1024 --gpu-warmup 5 --gpu-repeat 20
```

The scripts validate GPU results against CPU results. GPU timings exclude host
to device transfers and one-time CUDA JIT compilation.

## C And CUDA Baselines

Requirements:

- GCC or Clang with OpenMP support for CPU baselines
- NVIDIA CUDA Toolkit and `nvcc` for GPU baselines

Build and run a baseline from its own directory:

```bash
# Example: SpMV CPU
make -C GPU_CPU/SpMV/baselines/cpu
GPU_CPU/SpMV/baselines/cpu/spmv \
  -f GPU_CPU/SpMV/data/bcsstk30.mtx

# Example: SpMV GPU
make -C GPU_CPU/SpMV/baselines/gpu
GPU_CPU/SpMV/baselines/gpu/spmv \
  -f GPU_CPU/SpMV/data/bcsstk30.mtx
```

Equivalent `baselines/cpu` and `baselines/gpu` directories are provided for
GEMV, HIST, SRED, and VA. Their local `README` and `Makefile` files document
the target names and workload-specific options. GEMV (`gemv`), HIST
(`hist` on CPU, `hsti` on GPU), and VA (`va`) build with a plain `make`.


## Input Data

- `SpMV/data/bcsstk30.mtx` is the default Matrix Market sparse matrix.
- `HIST/input/image_VanHateren.iml` is the default histogram image.
- Other benchmarks generate their inputs at runtime.

These input files are benchmark data rather than build products.


