# CUDA MatMul ALgorithms Comparison

## Method Progression

- `cublas.cu`: Uses cuBLAS `cublasSgemm` (library GEMM kernel) for highly optimized matrix multiplication.
- `starter.cu`: Baseline naive matrix multiplication kernel (one thread computes one output element, direct global-memory accesses in inner loop).
- `v1.cu`: Introduces shared-memory tiling (`TILE=32`) with a 1D thread mapping, reducing redundant global-memory loads and improving data reuse.
- `v2.cu`: Moves to 2D block tiling with per-thread register blocking (`R=4`) plus shared-memory staging, increasing arithmetic intensity and reducing memory traffic further.
- `v3.cu`: Adds vectorized `float4` cooperative loads/stores for global-memory transfers in the 2D tiled + register-blocked kernel, improving memory throughput.

## NCU Results Summary

| Version      | Kernel                                            | Avg Kernel Time (ms) | GFLOPS   | Perf vs `cublas.cu` |
| ------------ | ------------------------------------------------- | -------------------- | -------- | ------------------- |
| `starter.cu` | `matmul_kernel_naive`                             | 104.953              | 1309.52  | 8.72%               |
| `v1.cu`      | `matmul_kernel_1d_tiled`                          | 88.803               | 1547.68  | 10.31%              |
| `v2.cu`      | `matmul_kernel_2d_block_tiled<4>`                 | 16.877               | 8143.73  | 54.23%              |
| `v3.cu`      | `matmul_kernel_2d_block_tiled<4>`                 | 13.080               | 10507.57 | 69.98%              |
| `cublas.cu`  | `Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn...` | 9.153                | 15015.18 | 100.00%             |
