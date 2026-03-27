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

// 2D block tiling + per-thread register blocking.
// Each block computes a TILE x TILE output tile.
// Each thread computes an R x R sub-tile in registers.
constexpr int TILE = 32;
constexpr int R = 4;

template <int REG_TILE>
__global__ void matmul_kernel_2d_block_tiled(const float *__restrict__ A,
                                             const float *__restrict__ B,
                                             float *__restrict__ C, int N) {
  extern __shared__ float smem[];
  float *As = smem;
  float *Bs = smem + TILE * TILE;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int local_row_base = ty * REG_TILE;
  const int local_col_base = tx * REG_TILE;
  const int row_base = blockIdx.y * TILE + local_row_base;
  const int col_base = blockIdx.x * TILE + local_col_base;

  float reg[REG_TILE][REG_TILE];
#pragma unroll
  for (int i = 0; i < REG_TILE; ++i) {
#pragma unroll
    for (int j = 0; j < REG_TILE; ++j) {
      reg[i][j] = 0.0f;
    }
  }

  const int num_tiles = (N + TILE - 1) / TILE;

  for (int t = 0; t < num_tiles; ++t) {
    // Cooperative load of A and B tiles: each thread loads REG_TILE x REG_TILE.
#pragma unroll
    for (int i = 0; i < REG_TILE; ++i) {
#pragma unroll
      for (int j = 0; j < REG_TILE; ++j) {
        const int a_row = row_base + i;
        const int a_col = t * TILE + local_col_base + j;
        As[(local_row_base + i) * TILE + (local_col_base + j)] =
            (a_row < N && a_col < N) ? A[a_row * N + a_col] : 0.0f;

        const int b_row = t * TILE + local_row_base + i;
        const int b_col = col_base + j;
        Bs[(local_row_base + i) * TILE + (local_col_base + j)] =
            (b_row < N && b_col < N) ? B[b_row * N + b_col] : 0.0f;
      }
    }

    __syncthreads();

// Multiply the shared-memory tiles into register accumulators.
#pragma unroll
    for (int k = 0; k < TILE; ++k) {
#pragma unroll
      for (int i = 0; i < REG_TILE; ++i) {
        const float a_val = As[(local_row_base + i) * TILE + k];
#pragma unroll
        for (int j = 0; j < REG_TILE; ++j) {
          reg[i][j] += a_val * Bs[k * TILE + (local_col_base + j)];
        }
      }
    }

    __syncthreads();
  }

// Write back this thread's REG_TILE x REG_TILE output tile.
#pragma unroll
  for (int i = 0; i < REG_TILE; ++i) {
#pragma unroll
    for (int j = 0; j < REG_TILE; ++j) {
      const int out_row = row_base + i;
      const int out_col = col_base + j;
      if (out_row < N && out_col < N) {
        C[out_row * N + out_col] = reg[i][j];
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
    dim3 block(TILE / R, TILE / R);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    size_t shared_bytes = 2 * TILE * TILE * sizeof(float);

    matmul_kernel_2d_block_tiled<R>
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

  // ── Performance Benchmark (kernel launches for ncu profiling) ──
  {
    const int N = 4096;
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

  return EXIT_SUCCESS;
}