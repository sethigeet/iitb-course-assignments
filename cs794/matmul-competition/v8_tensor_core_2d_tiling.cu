#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <mma.h>
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

namespace wmma = nvcuda::wmma;

// 2D tensor-core tiling: 8 warps arranged as a 4x2 warp grid.
// Each warp reuses one A fragment against two B fragments, so a 256-thread
// block computes a 64x64 output tile with better shared-memory reuse than v7.
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;
constexpr int BLOCK_M = WARPS_M * WMMA_M;
constexpr int BLOCK_N = WARPS_N * 2 * WMMA_N;
constexpr int BLOCK_DIM_X = WARP_SIZE;
constexpr int BLOCK_DIM_Y = WARPS_PER_BLOCK;
constexpr int THREADS_PER_BLOCK = BLOCK_DIM_X * BLOCK_DIM_Y;
constexpr int A_STRIDE = WMMA_K + 1;
constexpr int B_STRIDE = BLOCK_N + 8;

__global__ void matmul_kernel_tensor_core_2d_tiled(const float *__restrict__ A,
                                                   const float *__restrict__ B,
                                                   float *__restrict__ C,
                                                   int N) {
  extern __shared__ float smem[];
  float *As = smem;
  float *Bs = smem + BLOCK_M * A_STRIDE;

  const int lane_id = threadIdx.x;
  const int warp_id = threadIdx.y;
  const int tid = warp_id * WARP_SIZE + lane_id;
  const int warp_row = warp_id / WARPS_N;
  const int warp_col_pair = warp_id % WARPS_N;
  const int row_base = blockIdx.y * BLOCK_M;
  const int col_base = blockIdx.x * BLOCK_N;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag0;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag1;
  wmma::fill_fragment(c_frag0, 0.0f);
  wmma::fill_fragment(c_frag1, 0.0f);

  for (int k0 = 0; k0 < N; k0 += WMMA_K) {
    for (int idx = tid; idx < BLOCK_M * WMMA_K; idx += THREADS_PER_BLOCK) {
      const int r = idx / WMMA_K;
      const int c = idx % WMMA_K;
      As[r * A_STRIDE + c] = A[(row_base + r) * N + (k0 + c)];
    }
    for (int idx = tid; idx < WMMA_K * BLOCK_N; idx += THREADS_PER_BLOCK) {
      const int r = idx / BLOCK_N;
      const int c = idx % BLOCK_N;
      Bs[r * B_STRIDE + c] = B[(k0 + r) * N + (col_base + c)];
    }

    __syncthreads();

    const float *a_tile = &As[(warp_row * WMMA_M) * A_STRIDE];
    const int b_col0 = (warp_col_pair * 2 + 0) * WMMA_N;
    const int b_col1 = (warp_col_pair * 2 + 1) * WMMA_N;
    const float *b_tile0 = &Bs[b_col0];
    const float *b_tile1 = &Bs[b_col1];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        b_frag0;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        b_frag1;

    wmma::load_matrix_sync(a_frag, a_tile, A_STRIDE);
    wmma::load_matrix_sync(b_frag0, b_tile0, B_STRIDE);
    wmma::load_matrix_sync(b_frag1, b_tile1, B_STRIDE);
    wmma::mma_sync(c_frag0, a_frag, b_frag0, c_frag0);
    wmma::mma_sync(c_frag1, a_frag, b_frag1, c_frag1);

    __syncthreads();
  }

  const int c_row = row_base + warp_row * WMMA_M;
  const int c_col0 = col_base + (warp_col_pair * 2 + 0) * WMMA_N;
  const int c_col1 = col_base + (warp_col_pair * 2 + 1) * WMMA_N;
  wmma::store_matrix_sync(C + c_row * N + c_col0, c_frag0, N,
                          wmma::mem_row_major);
  wmma::store_matrix_sync(C + c_row * N + c_col1, c_frag1, N,
                          wmma::mem_row_major);
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
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  if (prop.major < 8) {
    fprintf(stderr,
            "Tensor-core TF32 kernel requires compute capability 8.0+, got %d.%d\n",
            prop.major, prop.minor);
    exit(EXIT_FAILURE);
  }

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
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (N + BLOCK_M - 1) / BLOCK_M);
    size_t shared_bytes =
        (BLOCK_M * A_STRIDE + WMMA_K * B_STRIDE) * sizeof(float);

    matmul_kernel_tensor_core_2d_tiled<<<grid, block, shared_bytes>>>(A_d, B_d,
                                                                      C_d, N);
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
    float limit = tol + 1e-3f * fabsf(ref[i]);
    if (diff > limit) {
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
  }

  return EXIT_SUCCESS;
}
