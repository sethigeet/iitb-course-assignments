# CUDA MatMul ALgorithms Comparison

## Method Progression

- `cublas.cu`: Uses cuBLAS `cublasSgemm` (library GEMM kernel) for highly optimized matrix multiplication.
- `starter.cu`: Baseline naive matrix multiplication kernel (one thread computes one output element, direct global-memory accesses in inner loop).
- `v1_1d_tiling.cu`: Introduces shared-memory tiling (`TILE=32`) with a 1D thread mapping, reducing redundant global-memory loads and improving data reuse.
- `v2_2d_tiling.cu`: Moves to 2D block tiling with per-thread register blocking (`R=4`) plus shared-memory staging, increasing arithmetic intensity and reducing memory traffic further.
- `v3_vectorized_loads.cu`: Adds vectorized `float4` cooperative loads/stores for global-memory transfers in the 2D tiled + register-blocked kernel, improving memory throughput.
- `v3.2_double_buffered.cu`: Splits the K loop into 32-wide shared-memory stages and ping-pongs two vectorized staging pages so the next stage is loaded before the current one is consumed.
- `v4_bank_conflicts.cu`: Pads only the `A` shared-memory tile rows with one extra column to reduce `A`-side shared-memory bank conflicts while keeping `B` tightly packed for vectorized staging.
- `v5_3d_tiling.cu`: Adds warp-level tiling on top of block tiling and pads only the `A` shared-memory tile to avoid the severe warp-stage shared-memory bank conflicts from the naive layout.
- `v5.2_3d_tiling_double_buffered.cu`: Splits `v5`'s K loop into 32-wide warp-tiled shared-memory stages and ping-pongs two pages so the next stage is staged while the current one is consumed.
- `v6_tensor_cores.cu`: Minimal TF32 WMMA kernel where one warp computes one 16x16 output tile, serving as the tensor-core baseline.
- `v7_tensor_core_1d_tiling.cu`: Uses 1D thread mapping with 8 warps per block to stage a 32x64 tile in shared memory and feed one WMMA tile per warp.
- `v8_tensor_core_2d_tiling.cu`: Reorganizes the tensor-core block into a 2D warp grid and lets each warp reuse one `A` fragment against two `B` fragments to cover a 64x64 tile.
- `v9_tensor_core_multistage.cu`: Keeps the 64x64 output tile but widens the shared-memory K stage to 32 so each block fill amortizes four WMMA steps.
- `v10_tensor_core_double_buffered.cu`: Splits the wider K stage into two shared-memory pages and software-pipelines fragment loads across them.
- `v11_tensor_core_warp_reuse.cu`: Has each warp compute a 32x32 output region (2x2 WMMA tiles) inside a 64x128 block tile to increase A/B fragment reuse.

## Build Notes

- `run.sh` now compiles with `nvcc -arch=native`, so the same source should build on both Ada GPUs such as the RTX A6000 Ada (`sm_89`) and newer GPUs such as the RTX 5060 Ti in this workspace.
- `run.sh` also accepts an optional benchmark size as a positional argument: `./run.sh file.cu 32768` or `./run.sh file.cu ncu 32768`.

## NCU Results Summary

### Small-N Results (`N=4096`)

| Version                             | Avg Kernel Time (ms) | GFLOPS   | Perf vs `cublas.cu` |
| ----------------------------------- | -------------------- | -------- | ------------------- |
| `starter.cu`                        | 104.953              | 1309.52  | 8.72%               |
| `v1_1d_tiling.cu`                   | 88.803               | 1547.68  | 10.31%              |
| `v2_2d_tiling.cu`                   | 16.877               | 8143.73  | 54.23%              |
| `v3_vectorized_loads.cu`            | 13.080               | 10507.57 | 69.98%              |
| `v3.2_double_buffered.cu`           | 14.870               | 9242.70  | 61.55%              |
| `v4_bank_conflicts.cu`              | 13.600               | 10105.81 | 67.30%              |
| `v5_3d_tiling.cu`                   | 13.483               | 10193.25 | 67.89%              |
| `v5.2_3d_tiling_double_buffered.cu` | 24.643               | 5577.13  | 37.14%              |
| `v6_tensor_cores.cu`                | 105.997              | 1296.63  | 8.64%               |
| `v7_tensor_core_1d_tiling.cu`       | 24.547               | 5599.09  | 37.29%              |
| `v8_tensor_core_2d_tiling.cu`       | 18.563               | 7403.79  | 49.31%              |
| `cublas.cu`                         | 9.153                | 15015.18 | 100.00%             |

### Large-N Results (`N=32768`)

| Version                              | Avg Kernel Time (ms) | GFLOPS   | Perf vs `cublas.cu` |
| ------------------------------------ | -------------------- | -------- | ------------------- |
| `v3_vectorized_loads.cu`             | 12973.333            | 5424.11  | 33.40%              |
| `v3.2_double_buffered.cu`            | 7580.000             | 9283.48  | 57.17%              |
| `v5_3d_tiling.cu`                    | 10603.333            | 6636.47  | 40.87%              |
| `v5.2_3d_tiling_double_buffered.cu`  | 12820.000            | 5488.98  | 33.80%              |
| `v9_tensor_core_multistage.cu`       | 7410.000             | 9496.46  | 58.48%              |
| `v10_tensor_core_double_buffered.cu` | 10260.000            | 6858.55  | 42.24%              |
| `v11_tensor_core_warp_reuse.cu`      | 7410.000             | 9496.46  | 58.48%              |
| `cublas.cu`                          | 4333.333             | 16238.94 | 100.00%             |
