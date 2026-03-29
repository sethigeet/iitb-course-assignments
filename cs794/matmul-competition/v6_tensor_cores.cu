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

// Minimal tensor-core kernel: one warp computes one 16x16 output tile.
// We use TF32 inputs so the same source runs on Ada (sm_89) and newer GPUs
// without requiring host-side float->half conversion kernels.
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;
constexpr int WARP_SIZE = 32;

__global__ void matmul_kernel_tensor_core_naive(const float *__restrict__ A,
                                                const float *__restrict__ B,
                                                float *__restrict__ C, int N) {
  extern __shared__ float smem[];
  float *As = smem;
  float *Bs = smem + WMMA_M * WMMA_K;

  const int lane_id = threadIdx.x;
  const int row_base = blockIdx.y * WMMA_M;
  const int col_base = blockIdx.x * WMMA_N;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < N; k0 += WMMA_K) {
    for (int idx = lane_id; idx < WMMA_M * WMMA_K; idx += WARP_SIZE) {
      const int r = idx / WMMA_K;
      const int c = idx % WMMA_K;
      As[idx] = A[(row_base + r) * N + (k0 + c)];
    }
    for (int idx = lane_id; idx < WMMA_K * WMMA_N; idx += WARP_SIZE) {
      const int r = idx / WMMA_N;
      const int c = idx % WMMA_N;
      Bs[idx] = B[(k0 + r) * N + (col_base + c)];
    }

    __syncthreads();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        b_frag;

    wmma::load_matrix_sync(a_frag, As, WMMA_K);
    wmma::load_matrix_sync(b_frag, Bs, WMMA_N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    __syncthreads();
  }

  wmma::store_matrix_sync(C + row_base * N + col_base, c_frag, N,
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
    dim3 block(WARP_SIZE);
    dim3 grid((N + WMMA_N - 1) / WMMA_N, (N + WMMA_M - 1) / WMMA_M);
    size_t shared_bytes = 2 * WMMA_M * WMMA_K * sizeof(float);

    matmul_kernel_tensor_core_naive<<<grid, block, shared_bytes>>>(A_d, B_d,
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