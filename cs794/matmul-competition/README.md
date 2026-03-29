# CUDA MatMul ALgorithms Comparison

## Method Progression

- `cublas.cu`: Uses cuBLAS `cublasSgemm` (library GEMM kernel) for highly optimized matrix multiplication.
- `starter.cu`: Baseline naive matrix multiplication kernel (one thread computes one output element, direct global-memory accesses in inner loop).
- `v1_1d_tiling.cu`: Introduces shared-memory tiling (`TILE=32`) with a 1D thread mapping, reducing redundant global-memory loads and improving data reuse.
- `v2_2d_tiling.cu`: Moves to 2D block tiling with per-thread register blocking (`R=4`) plus shared-memory staging, increasing arithmetic intensity and reducing memory traffic further.
- `v3_vectorized_loads.cu`: Adds vectorized `float4` cooperative loads/stores for global-memory transfers in the 2D tiled + register-blocked kernel, improving memory throughput.

## NCU Results Summary

| Version                  | Avg Kernel Time (ms) | GFLOPS   | Perf vs `cublas.cu` |
| ------------------------ | -------------------- | -------- | ------------------- |
| `starter.cu`             | 104.953              | 1309.52  | 8.72%               |
| `v1_1d_tiling.cu`        | 88.803               | 1547.68  | 10.31%              |
| `v2_2d_tiling.cu`        | 16.877               | 8143.73  | 54.23%              |
| `v3_vectorized_loads.cu` | 13.080               | 10507.57 | 69.98%              |
| `cublas.cu`              | 9.153                | 15015.18 | 100.00%             |
