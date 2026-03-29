#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

// ─── Error-checking macro ────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d  %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/* ============================================================
   Edit ONLY this section.
   ============================================================ */

// v5 + double buffering: keep the 64x64 block / 32x16 warp / 4x4 thread
// hierarchy, but split the K loop into 32-wide shared-memory stages so two
// pages can ping-pong while the warp-level microkernel consumes the current one.
constexpr int TILE = 64;
constexpr int STAGE_K = 32;
constexpr int NUM_STAGES = 2;
constexpr int THREAD_TILE_M = 4;
constexpr int THREAD_TILE_N = 4;
constexpr int WARP_TILE_M = 32;
constexpr int WARP_TILE_N = 16;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_M = TILE / WARP_TILE_M;
constexpr int WARPS_N = TILE / WARP_TILE_N;
constexpr int THREADS_PER_BLOCK = WARPS_M * WARPS_N * WARP_SIZE;
constexpr int BLOCK_DIM_X = 16;
constexpr int BLOCK_DIM_Y = THREADS_PER_BLOCK / BLOCK_DIM_X;
constexpr int A_STAGE_STRIDE = STAGE_K + 1;
constexpr int B_STAGE_STRIDE = TILE;
constexpr int A_PAGE_FLOATS = TILE * A_STAGE_STRIDE;
constexpr int B_PAGE_FLOATS = STAGE_K * B_STAGE_STRIDE;

static_assert(WARPS_M * WARPS_N == 8, "This kernel expects 8 warps per block.");
static_assert(THREADS_PER_BLOCK == 256, "This kernel expects 256 threads.");
static_assert(STAGE_K % 4 == 0, "Stage width must stay float4 aligned.");
static_assert((WARP_TILE_M / THREAD_TILE_M) * (WARP_TILE_N / THREAD_TILE_N) ==
                  WARP_SIZE,
              "Warp tile must map cleanly to 32 threads.");

__device__ inline void load_stage_page(const float *__restrict__ A,
                                       const float *__restrict__ B, float *As,
                                       float *Bs, int N, int tile_idx, int tid,
                                       int threads_per_block) {
  constexpr int VEC = 4;
  constexpr int A_STAGE_VECS = STAGE_K / VEC;
  constexpr int B_TILE_VECS = TILE / VEC;

  for (int idx = tid; idx < TILE * A_STAGE_VECS; idx += threads_per_block) {
    const int inner_row = idx / A_STAGE_VECS;
    const int inner_col = idx % A_STAGE_VECS;
    const int col4 = inner_col * VEC;

    const int a_row = blockIdx.y * TILE + inner_row;
    const int a_col = tile_idx * STAGE_K + col4;
    if (a_row < N && a_col + (VEC - 1) < N) {
      const float4 a_vec =
          reinterpret_cast<const float4 *>(&A[a_row * N + a_col])[0];
      As[inner_row * A_STAGE_STRIDE + col4 + 0] = a_vec.x;
      As[inner_row * A_STAGE_STRIDE + col4 + 1] = a_vec.y;
      As[inner_row * A_STAGE_STRIDE + col4 + 2] = a_vec.z;
      As[inner_row * A_STAGE_STRIDE + col4 + 3] = a_vec.w;
    } else {
#pragma unroll
      for (int v = 0; v < VEC; ++v) {
        const int g_col = a_col + v;
        As[inner_row * A_STAGE_STRIDE + col4 + v] =
            (a_row < N && g_col < N) ? A[a_row * N + g_col] : 0.0f;
      }
    }
  }

  for (int idx = tid; idx < STAGE_K * B_TILE_VECS; idx += threads_per_block) {
    const int inner_row = idx / B_TILE_VECS;
    const int inner_col = idx % B_TILE_VECS;
    const int col4 = inner_col * VEC;

    const int b_row = tile_idx * STAGE_K + inner_row;
    const int b_col = blockIdx.x * TILE + col4;
    if (b_row < N && b_col + (VEC - 1) < N) {
      reinterpret_cast<float4 *>(&Bs[inner_row * B_STAGE_STRIDE + col4])[0] =
          reinterpret_cast<const float4 *>(&B[b_row * N + b_col])[0];
    } else {
#pragma unroll
      for (int v = 0; v < VEC; ++v) {
        const int g_col = b_col + v;
        Bs[inner_row * B_STAGE_STRIDE + col4 + v] =
            (b_row < N && g_col < N) ? B[b_row * N + g_col] : 0.0f;
      }
    }
  }
}

template <int TM, int TN>
__global__ void matmul_kernel_3d_warp_tiled_double_buffered(
    const float *__restrict__ A, const float *__restrict__ B,
    float *__restrict__ C, int N) {
  extern __shared__ float smem[];
  float *As_pages = smem;
  float *Bs_pages = smem + NUM_STAGES * A_PAGE_FLOATS;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int threads_per_block = blockDim.x * blockDim.y;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int warp_row = warp_id / WARPS_N;
  const int warp_col = warp_id % WARPS_N;
  const int lane_row = lane_id / (WARP_TILE_N / TN);
  const int lane_col = lane_id % (WARP_TILE_N / TN);

  const int warp_row_base = warp_row * WARP_TILE_M;
  const int warp_col_base = warp_col * WARP_TILE_N;
  const int thread_row_base = warp_row_base + lane_row * TM;
  const int thread_col_base = warp_col_base + lane_col * TN;
  const int row_base = blockIdx.y * TILE + thread_row_base;
  const int col_base = blockIdx.x * TILE + thread_col_base;

  float thread_results[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      thread_results[i][j] = 0.0f;
    }
  }

  const int num_tiles = (N + STAGE_K - 1) / STAGE_K;
  constexpr int VEC = 4;

  load_stage_page(A, B, As_pages, Bs_pages, N, 0, tid, threads_per_block);
  __syncthreads();

  for (int t = 0; t < num_tiles; ++t) {
    const int stage = t & 1;
    const int next_stage = stage ^ 1;
    float *As = As_pages + stage * A_PAGE_FLOATS;
    float *Bs = Bs_pages + stage * B_PAGE_FLOATS;

    if (t + 1 < num_tiles) {
      float *As_next = As_pages + next_stage * A_PAGE_FLOATS;
      float *Bs_next = Bs_pages + next_stage * B_PAGE_FLOATS;
      load_stage_page(A, B, As_next, Bs_next, N, t + 1, tid, threads_per_block);
    }

    // Each warp stages one 32x16 tile from the active shared-memory page.
#pragma unroll
    for (int k = 0; k < STAGE_K; ++k) {
      float regM[TM];
      float regN[TN];

#pragma unroll
      for (int i = 0; i < TM; ++i) {
        regM[i] = As[(thread_row_base + i) * A_STAGE_STRIDE + k];
      }

#pragma unroll
      for (int j = 0; j < TN; ++j) {
        regN[j] = Bs[k * B_STAGE_STRIDE + thread_col_base + j];
      }

#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          thread_results[i][j] += regM[i] * regN[j];
        }
      }
    }

    __syncthreads();
  }

  // Write back this thread's 4x4 micro-tile.
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int out_row = row_base + i;
    const int out_col = col_base;
    if (out_row < N && out_col + (VEC - 1) < N) {
      float4 out = {thread_results[i][0], thread_results[i][1],
                    thread_results[i][2], thread_results[i][3]};
      reinterpret_cast<float4 *>(&C[out_row * N + out_col])[0] = out;
    } else {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int c_col = out_col + j;
        if (out_row < N && c_col < N) {
          C[out_row * N + c_col] = thread_results[i][j];
        }
      }
    }
  }
}

/**
 * @brief Launch wrapper — allocate device memory, copy data,
 *        run your kernel(s), copy result back. You aren't allowed to change
 * this function signature.
 *
 * @param N    Matrix dimension (N x N).  Always a power of 2.
 * @param A_h  Host pointer to matrix A (row-major, N*N floats).
 * @param B_h  Host pointer to matrix B (row-major, N*N floats).
 * @param C_h  Host pointer to output C (row-major, N*N floats).
 *             You must write the result here before returning.
 */
void matmul_gpu(int N, const float *A_h, const float *B_h, float *C_h) {
  size_t bytes = (size_t)N * N * sizeof(float);

  // ── Allocate device buffers ───────────────────────────────
  float *A_d, *B_d, *C_d;
  CUDA_CHECK(cudaMalloc(&A_d, bytes));
  CUDA_CHECK(cudaMalloc(&B_d, bytes));
  CUDA_CHECK(cudaMalloc(&C_d, bytes));

  // ── Transfer inputs to device ─────────────────────────────
  CUDA_CHECK(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice));

  {
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    size_t shared_bytes =
        NUM_STAGES * (A_PAGE_FLOATS + B_PAGE_FLOATS) * sizeof(float);

    matmul_kernel_3d_warp_tiled_double_buffered<THREAD_TILE_M, THREAD_TILE_N>
        <<<grid, block, shared_bytes>>>(A_d, B_d, C_d, N);
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  // ── Copy result back to host ──────────────────────────────
  CUDA_CHECK(cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost));

  // ── Free device memory ────────────────────────────────────
  CUDA_CHECK(cudaFree(A_d));
  CUDA_CHECK(cudaFree(B_d));
  CUDA_CHECK(cudaFree(C_d));
}

/* ============================================================
   END OF STUDENT CODE — do not modify below this line
   ============================================================ */

// ─── CPU reference
// ────────────────────────────────────────────────────────────
static void matmul_cpu(int N, const float *A, const float *B, float *C) {
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      float s = 0.0f;
      for (int k = 0; k < N; ++k)
        s += A[i * N + k] * B[k * N + j];
      C[i * N + j] = s;
    }
}

// ─── Element-wise verification
// ────────────────────────────────────────────────
static bool verify(int N, const float *ref, const float *gpu,
                   float tol = 1e-2f) {
  for (int i = 0; i < N * N; ++i) {
    float diff = fabsf(ref[i] - gpu[i]);
    if (diff > tol) {
      int row = i / N, col = i % N;
      fprintf(stderr, "MISMATCH at (%d,%d): ref=%.6f  gpu=%.6f  |diff|=%.2e\n",
              row, col, ref[i], gpu[i], diff);
      return false;
    }
  }
  return true;
}

// ─── main
// ─────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
  bool benchmark_only = (argc > 1 && strcmp(argv[1], "benchmark") == 0);
  bool correctness_only = (argc > 1 && strcmp(argv[1], "correctness") == 0);

  if (!benchmark_only) {
    // ── Correctness tests (small sizes, CPU reference) ────────
    printf("=== Correctness Tests ===\n");
    {
      const std::vector<int> small_sizes = {64, 128, 256, 512};
      bool all_ok = true;

      for (int N : small_sizes) {

        std::vector<float> A(N * N), B(N * N), C_cpu(N * N, 0.f),
            C_gpu(N * N, 0.f);

        for (int i = 0; i < N * N; ++i) {
          A[i] = (float)(i % 97) / 97.f;
          B[i] = (float)((i * 7 + 3) % 97) / 97.f;
        }

        matmul_cpu(N, A.data(), B.data(), C_cpu.data());
        matmul_gpu(N, A.data(), B.data(), C_gpu.data());

        bool ok = verify(N, C_cpu.data(), C_gpu.data());
        printf("  N = %4d : %s\n", N, ok ? "PASSED" : "FAILED");
        all_ok &= ok;
      }

      if (!all_ok) {
        fprintf(stderr,
                "\nCorrectness FAILED — fix your kernel before optimising.\n");
        return EXIT_FAILURE;
      }
      printf("All correctness tests PASSED.\n\n");
    }
  } // !benchmark_only

  if (!correctness_only) {
    // ── Performance Benchmark (kernel launches for ncu profiling) ──
    {
      int N = 4096;
      if (argc > 2) {
        const char *value = argv[2];
        char *end = nullptr;
        long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || parsed <= 0) {
          fprintf(stderr, "Invalid benchmark size: %s\n", value);
          return EXIT_FAILURE;
        }
        N = static_cast<int>(parsed);
      }
      const int NUM_RUNS = 3;
      printf("=== Performance Benchmark (N=%d, 1 warmup + %d runs) ===\n", N,
             NUM_RUNS);

      size_t elems = (size_t)N * N;
      std::vector<float> A(elems), B(elems), C(elems);
      for (size_t i = 0; i < elems; ++i) {
        A[i] = (float)(i % 97) / 97.f;
        B[i] = (float)((i * 7 + 3) % 97) / 97.f;
      }

      matmul_gpu(N, A.data(), B.data(), C.data());
      printf("  Warmup complete.\n");

      for (int r = 0; r < NUM_RUNS; ++r)
        matmul_gpu(N, A.data(), B.data(), C.data());

      printf("  %d measured runs complete.\n", NUM_RUNS);
    }
  }

  return EXIT_SUCCESS;
}
