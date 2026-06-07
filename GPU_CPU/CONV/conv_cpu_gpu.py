from __future__ import annotations

import argparse
import platform
import time

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
# Custom CUDA kernel: tiled 1-D convolution with shared memory
#
# Design:
#   - Each thread block loads a contiguous tile of the padded signal into
#     shared memory (BLOCK*stride + kernel_size - 1 floats), so every
#     global-memory read is done exactly once per block.
#   - Each thread computes one output element using FMA (__fmaf_rn).
#   - Striding is handled directly inside the kernel (no extra pass).
#   - The kernel index is reversed to match np.convolve (true convolution,
#     not cross-correlation).
# ---------------------------------------------------------------------------

_CONV1D_SRC = r"""
extern "C" __global__
__launch_bounds__(256)
void conv1d_direct(
    const float* __restrict__ padded,
    const float* __restrict__ kernel_w,
    float*       __restrict__ output,
    const int padded_len,
    const int kernel_size,
    const int stride,
    const int output_len
) {
    extern __shared__ float s_tile[];

    const int block_out_start = blockIdx.x * blockDim.x;
    const int block_in_start  = block_out_start * stride;
    const int tile_size = blockDim.x * stride + kernel_size - 1;

    // Cooperative load into shared memory
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
        const int gi = block_in_start + i;
        s_tile[i] = (gi < padded_len) ? padded[gi] : 0.0f;
    }
    __syncthreads();

    const int out_idx = block_out_start + threadIdx.x;
    if (out_idx < output_len) {
        float acc = 0.0f;
        const int local_start = threadIdx.x * stride;
        for (int k = 0; k < kernel_size; k++) {
            acc = __fmaf_rn(s_tile[local_start + k], kernel_w[kernel_size - 1 - k], acc);
        }
        output[out_idx] = acc;
    }
}
"""

_raw_kernel = None


def _get_raw_kernel():
	global _raw_kernel
	if _raw_kernel is None:
		_raw_kernel = cp.RawKernel(_CONV1D_SRC, "conv1d_direct")
	return _raw_kernel


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT_SIZE = 1_000_000
DEFAULT_KERNEL_SIZE = 10
DEFAULT_STRIDE = 2
DEFAULT_PADDING = 4
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

def build_inputs(input_size: int, kernel_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
	rng = np.random.default_rng(seed)
	signal = rng.random(input_size, dtype=np.float32)
	kernel = rng.random(kernel_size, dtype=np.float32)
	return signal, kernel


def conv1d_cpu(signal: np.ndarray, kernel: np.ndarray, stride: int, padding: int) -> np.ndarray:
	padded = np.pad(signal, (padding, padding), mode="constant")
	full = np.convolve(padded, kernel, mode="valid")
	return full[::stride]


def _cuda_timed(fn):
	"""Run fn() bracketed by CUDA events; return (result, elapsed_seconds)."""
	t0 = cp.cuda.Event()
	t1 = cp.cuda.Event()
	t0.record()
	result = fn()
	t1.record()
	t1.synchronize()
	return result, cp.cuda.get_elapsed_time(t0, t1) / 1000.0


def conv1d_gpu(
	signal: np.ndarray, kernel: np.ndarray, stride: int, padding: int
) -> tuple[np.ndarray, float, float]:
	"""
	Returns (output, cupy_elapsed_s, custom_elapsed_s).
	Both timings are compute-only (exclude host<->device transfer and JIT).
	"""
	if cp is None:
		raise RuntimeError("CuPy is not installed.")
	if cp.cuda.runtime.getDeviceCount() == 0:
		raise RuntimeError("No CUDA-capable GPU is available.")

	signal_gpu = cp.asarray(signal)
	kernel_gpu = cp.asarray(kernel)
	padded_gpu = cp.pad(signal_gpu, (padding, padding), mode="constant")
	padded_len = int(padded_gpu.size)
	kernel_size = int(kernel_gpu.size)
	output_len = (padded_len - kernel_size) // stride + 1

	# ---- Baseline: cp.convolve (FFT-based internally) ----
	_ = cp.convolve(padded_gpu[: kernel_size + stride], kernel_gpu, mode="valid")
	cp.cuda.Stream.null.synchronize()

	_, cupy_elapsed = _cuda_timed(
		lambda: cp.convolve(padded_gpu, kernel_gpu, mode="valid")[::stride]
	)

	# ---- Optimized: tiled shared-memory direct-convolution kernel ----
	BLOCK = 256
	smem_bytes = (BLOCK * stride + kernel_size - 1) * 4  # float32
	grid = (output_len + BLOCK - 1) // BLOCK
	output_gpu = cp.empty(output_len, dtype=cp.float32)

	raw = _get_raw_kernel()

	def _launch():
		raw(
			(grid,), (BLOCK,),
			(padded_gpu, kernel_gpu, output_gpu,
			 padded_len, kernel_size, stride, output_len),
			shared_mem=smem_bytes,
		)
		return output_gpu

	_launch()  # warmup: triggers PTX JIT compilation
	cp.cuda.Stream.null.synchronize()

	_, custom_elapsed = _cuda_timed(_launch)

	return cp.asnumpy(output_gpu), cupy_elapsed, custom_elapsed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="1D convolution benchmark on CPU and GPU")
	parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
	parser.add_argument("--kernel-size", type=int, default=DEFAULT_KERNEL_SIZE)
	parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
	parser.add_argument("--padding", type=int, default=DEFAULT_PADDING)
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.input_size <= 0:
		raise ValueError("input-size must be positive")
	if args.kernel_size <= 0:
		raise ValueError("kernel-size must be positive")
	if args.stride <= 0:
		raise ValueError("stride must be positive")
	if args.padding < 0:
		raise ValueError("padding must be non-negative")

	print(f"CPU: {get_cpu_info()}")
	print(f"GPU: {get_gpu_info()}")

	signal, kernel = build_inputs(args.input_size, args.kernel_size, args.seed)

	cpu_start = time.perf_counter()
	cpu_output = conv1d_cpu(signal, kernel, args.stride, args.padding)
	cpu_elapsed = time.perf_counter() - cpu_start

	gpu_output, cupy_elapsed, custom_elapsed = conv1d_gpu(
		signal, kernel, args.stride, args.padding
	)

	expected_output_size = (
		(args.input_size + 2 * args.padding - args.kernel_size) // args.stride
	) + 1

	print(f"Input size:               {args.input_size}")
	print(f"Kernel size:              {args.kernel_size}")
	print(f"Stride:                   {args.stride}")
	print(f"Padding:                  {args.padding}")
	print(f"Output size:              {expected_output_size}")
	print(f"CPU time:                 {cpu_elapsed * 1000:.6f} ms")
	print(f"GPU time (cp.convolve):   {cupy_elapsed * 1000:.6f} ms  [compute only]")
	print(f"GPU time (custom kernel): {custom_elapsed * 1000:.6f} ms  [compute only]")
	if cupy_elapsed > 0 and custom_elapsed > 0:
		print(f"Speedup custom vs cupy:   {cupy_elapsed / custom_elapsed:.1f}x")
	print(f"Results match:            {np.allclose(cpu_output, gpu_output, rtol=1e-5, atol=1e-5)}")


if __name__ == "__main__":
	main()
