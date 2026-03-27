# CUDA MatMul ALgorithms Comparison

## Method Progression

- `starter.cu`: Baseline naive matrix multiplication kernel (one thread computes one output element, direct global-memory accesses in inner loop).
- `v1.cu`: Introduces shared-memory tiling (`TILE=32`) with a 1D thread mapping, reducing redundant global-memory loads and improving data reuse.
- `v2.cu`: Moves to 2D block tiling with per-thread register blocking (`R=4`) plus shared-memory staging, increasing arithmetic intensity and reducing memory traffic further.

## NCU Results Summary


| Version      | Kernel                            | Avg Kernel Time (ms) | GFLOPS  | Speedup vs `starter.cu` | Improvement vs Previous             |
| ------------ | --------------------------------- | -------------------- | ------- | ----------------------- | ----------------------------------- |
| `starter.cu` | `matmul_kernel_naive`             | 104.953              | 1309.52 | 1.00x                   | -                                   |
| `v1.cu`      | `matmul_kernel_1d_tiled`          | 88.803               | 1547.68 | 1.18x                   | 15.39% lower time than `starter.cu` |
| `v2.cu`      | `matmul_kernel_2d_block_tiled<4>` | 25.557               | 5377.81 | 4.11x                   | 71.22% lower time than `v1.cu`      |


`v2.cu` is the fastest among the three, with a 4.11x speedup over baseline.