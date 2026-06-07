from __future__ import annotations

import argparse
import os
import platform
import time

# Restrict CPU thread pools to 16 cores BEFORE numpy is imported.
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("BLIS_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover - handled at runtime
    cp = None


# ---------------------------------------------------------------------------
# Device info
# ---------------------------------------------------------------------------

def get_cpu_info() -> str:
	try:
		with open("/proc/cpuinfo") as f:
			for line in f:
				if line.startswith("model name"):
					return line.split(":", 1)[1].strip()
	except OSError:
		pass
	return platform.processor() or "Unknown CPU"


def get_gpu_info() -> str:
	if cp is None:
		return "CuPy not installed"
	try:
		if cp.cuda.runtime.getDeviceCount() == 0:
			return "No CUDA GPU found"
		props = cp.cuda.runtime.getDeviceProperties(0)
		name = props["name"]
		return name.decode() if isinstance(name, bytes) else name
	except Exception as e:
		return f"Error querying GPU: {e}"


# ---------------------------------------------------------------------------
# Custom CUDA kernel: tiled GEMM with shared memory
#
# Design:
#   - Computes C = A @ B where A is MxK, B is KxN, C is MxN (row-major).
#   - Each thread block computes a TILE x TILE output sub-block.
#   - A and B tiles are cooperatively staged in shared memory; each global
#     memory element is read once per block.
#   - Inner accumulation uses FMA (__fmaf_rn) for full single-precision
#     fused multiply-add.
# ---------------------------------------------------------------------------

_GEMM_TILE = 32

_GEMM_SRC = r"""
extern "C" __global__
__launch_bounds__(1024)
void gemm_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int M,
    const int N,
    const int K
) {
    const int TILE = 32;
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    const int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; t++) {
        const int a_col = t * TILE + threadIdx.x;
        const int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc = __fmaf_rn(As[threadIdx.y][k], Bs[k][threadIdx.x], acc);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
"""

_raw_kernel = None


def _get_raw_kernel():
	global _raw_kernel
	if _raw_kernel is None:
		_raw_kernel = cp.RawKernel(_GEMM_SRC, "gemm_tiled")
	return _raw_kernel


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_M = 1024
DEFAULT_K = 1024
DEFAULT_N = 1024
# DEFAULT_M = 8192
# DEFAULT_K = 8192
# DEFAULT_N = 8192
DEFAULT_SEED = 42
DEFAULT_GPU_WARMUP = 5
DEFAULT_GPU_REPEAT = 20


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

def build_inputs(M: int, K: int, N: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
	rng = np.random.default_rng(seed)
	A = rng.random((M, K), dtype=np.float32)
	B = rng.random((K, N), dtype=np.float32)
	return A, B


def gemm_cpu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
	return A @ B


def _cuda_timed(fn):
	"""Run fn() bracketed by CUDA events; return (result, elapsed_ms)."""
	t0 = cp.cuda.Event()
	t1 = cp.cuda.Event()
	t0.record()
	result = fn()
	t1.record()
	t1.synchronize()
	return result, cp.cuda.get_elapsed_time(t0, t1)


def _cuda_benchmark(fn, warmup: int, repeat: int) -> float:
	"""Warm up fn(), then return the median elapsed time in ms."""
	for _ in range(warmup):
		fn()
	cp.cuda.Stream.null.synchronize()

	elapsed_ms = []
	for _ in range(repeat):
		_, one_elapsed_ms = _cuda_timed(fn)
		elapsed_ms.append(one_elapsed_ms)
	return float(np.median(elapsed_ms))


def gemm_gpu(
	A: np.ndarray,
	B: np.ndarray,
	warmup: int,
	repeat: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
	"""
	Returns (cupy_output, custom_output, cupy_elapsed_ms, custom_elapsed_ms).
	Both timings are compute-only (exclude host<->device transfer and JIT).
	"""
	if cp is None:
		raise RuntimeError("CuPy is not installed.")
	if cp.cuda.runtime.getDeviceCount() == 0:
		raise RuntimeError("No CUDA-capable GPU is available.")

	M, K = A.shape
	K2, N = B.shape
	assert K == K2, "Inner dimensions of A and B must match"

	A_gpu = cp.asarray(A)
	B_gpu = cp.asarray(B)

	# ---- Baseline: cp.matmul (cuBLAS) ----
	C_cupy = cp.empty((M, N), dtype=cp.float32)

	def _launch_cupy():
		return cp.matmul(A_gpu, B_gpu, out=C_cupy)

	# ---- Optimized: tiled shared-memory direct-GEMM kernel ----
	TILE = _GEMM_TILE
	C_custom = cp.empty((M, N), dtype=cp.float32)
	grid = ((N + TILE - 1) // TILE, (M + TILE - 1) // TILE)
	block = (TILE, TILE)

	raw = _get_raw_kernel()

	def _launch():
		raw(
			grid, block,
			(A_gpu, B_gpu, C_custom,
			 np.int32(M), np.int32(N), np.int32(K)),
		)
		return C_custom

	cupy_elapsed_ms = _cuda_benchmark(_launch_cupy, warmup, repeat)
	custom_elapsed_ms = _cuda_benchmark(_launch, warmup, repeat)

	return cp.asnumpy(C_cupy), cp.asnumpy(C_custom), cupy_elapsed_ms, custom_elapsed_ms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="GEMM benchmark on CPU and GPU")
	parser.add_argument("-M", type=int, default=DEFAULT_M, help="rows of A / rows of C")
	parser.add_argument("-K", type=int, default=DEFAULT_K, help="cols of A / rows of B")
	parser.add_argument("-N", type=int, default=DEFAULT_N, help="cols of B / cols of C")
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
	parser.add_argument("--gpu-warmup", type=int, default=DEFAULT_GPU_WARMUP)
	parser.add_argument("--gpu-repeat", type=int, default=DEFAULT_GPU_REPEAT)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.M <= 0 or args.K <= 0 or args.N <= 0:
		raise ValueError("M, K, N must be positive")
	if args.gpu_warmup < 0:
		raise ValueError("--gpu-warmup must be non-negative")
	if args.gpu_repeat <= 0:
		raise ValueError("--gpu-repeat must be positive")

	print(f"CPU: {get_cpu_info()}")
	print(f"GPU: {get_gpu_info()}")

	A, B = build_inputs(args.M, args.K, args.N, args.seed)

	# CPU warmup (BLAS thread-pool init) then timed run.
	_ = gemm_cpu(A[:64, :64], B[:64, :64])
	cpu_start = time.perf_counter()
	cpu_output = gemm_cpu(A, B)
	cpu_elapsed_ms = (time.perf_counter() - cpu_start) * 1000.0

	cupy_output, custom_output, cupy_elapsed_ms, custom_elapsed_ms = gemm_gpu(
		A, B, args.gpu_warmup, args.gpu_repeat
	)

	flops = 2.0 * args.M * args.N * args.K

	print(f"Matrix A:                 {args.M} x {args.K}")
	print(f"Matrix B:                 {args.K} x {args.N}")
	print(f"Matrix C (output):        {args.M} x {args.N}")
	print(f"CPU threads:              {os.environ.get('OMP_NUM_THREADS', 'default')}")
	print(f"GPU warmup/repeat:        {args.gpu_warmup} / {args.gpu_repeat}  [median]")
	print(f"CPU time:                 {cpu_elapsed_ms:.3f} ms  ({flops / cpu_elapsed_ms / 1e6:.2f} GFLOP/s)")
	print(f"GPU time (cp.matmul):     {cupy_elapsed_ms:.3f} ms  [compute only]  ({flops / cupy_elapsed_ms / 1e6:.2f} GFLOP/s)")
	print(f"GPU time (custom kernel): {custom_elapsed_ms:.3f} ms  [compute only]  ({flops / custom_elapsed_ms / 1e6:.2f} GFLOP/s)")
	if cupy_elapsed_ms > 0:
		print(f"Speedup CPU vs cp.matmul: {cpu_elapsed_ms / cupy_elapsed_ms:.2f}x")
	if custom_elapsed_ms > 0:
		print(f"Speedup CPU vs custom:    {cpu_elapsed_ms / custom_elapsed_ms:.2f}x")
	print(f"cp.matmul matches CPU:    {np.allclose(cpu_output, cupy_output, rtol=1e-3, atol=1e-3)}")
	print(f"custom matches CPU:       {np.allclose(cpu_output, custom_output, rtol=1e-3, atol=1e-3)}")


if __name__ == "__main__":
	main()
