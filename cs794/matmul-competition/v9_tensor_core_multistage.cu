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

// v9: widen the K stage so each shared-memory fill feeds four WMMA steps.
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int BLOCK_DIM_X = WARP_SIZE;
constexpr int BLOCK_DIM_Y = WARPS_PER_BLOCK;
constexpr int FRAGS_N_PER_WARP = 2;
constexpr int BLOCK_M = WARPS_M * WMMA_M;
constexpr int BLOCK_N = WARPS_N * FRAGS_N_PER_WARP * WMMA_N;
constexpr int BLOCK_K = 32;
constexpr int A_STRIDE = BLOCK_K + 1;
constexpr int B_STRIDE = BLOCK_N + 8;

static_assert(BLOCK_M == 64 && BLOCK_N == 64, "v9 is tuned for a 64x64 tile.");
static_assert(THREADS_PER_BLOCK == 256, "v9 expects 256 threads per block.");

__device__ inline void load_a_stage(const float *__restrict__ A, float *As,
                                    int row_base, int k_base, int N, int tid) {
  constexpr int VEC = 4;
  constexpr int VECS_PER_ROW = BLOCK_K / VEC;
  const int total_vecs = BLOCK_M * VECS_PER_ROW;
  for (int idx = tid; idx < total_vecs; idx += THREADS_PER_BLOCK) {
    const int row = idx / VECS_PER_ROW;
    const int col4 = (idx % VECS_PER_ROW) * VEC;
    const float4 a_vec =
        reinterpret_cast<const float4 *>(&A[(row_base + row) * N + k_base + col4])[0];
    As[row * A_STRIDE + col4 + 0] = a_vec.x;
    As[row * A_STRIDE + col4 + 1] = a_vec.y;
    As[row * A_STRIDE + col4 + 2] = a_vec.z;
    As[row * A_STRIDE + col4 + 3] = a_vec.w;
  }
}

__device__ inline void load_b_stage(const float *__restrict__ B, float *Bs,
                                    int col_base, int k_base, int N, int tid) {
  constexpr int VEC = 4;
  constexpr int VECS_PER_ROW = BLOCK_N / VEC;
  const int total_vecs = BLOCK_K * VECS_PER_ROW;
  for (int idx = tid; idx < total_vecs; idx += THREADS_PER_BLOCK) {
    const int row = idx / VECS_PER_ROW;
    const int col4 = (idx % VECS_PER_ROW) * VEC;
    reinterpret_cast<float4 *>(&Bs[row * B_STRIDE + col4])[0] =
        reinterpret_cast<const float4 *>(&B[(k_base + row) * N + col_base + col4])[0];
  }
}

__global__ void matmul_kernel_tensor_core_multistage(const float *__restrict__ A,
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
  const int warp_col_group = warp_id % WARPS_N;
  const int row_base = blockIdx.y * BLOCK_M;
  const int col_base = blockIdx.x * BLOCK_N;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[FRAGS_N_PER_WARP];
#pragma unroll
  for (int j = 0; j < FRAGS_N_PER_WARP; ++j) {
    wmma::fill_fragment(c_frag[j], 0.0f);
  }

  for (int k_base = 0; k_base < N; k_base += BLOCK_K) {
    load_a_stage(A, As, row_base, k_base, N, tid);
    load_b_stage(B, Bs, col_base, k_base, N, tid);
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
      const float *a_ptr = &As[(warp_row * WMMA_M) * A_STRIDE + kk];

      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          a_frag;
      wmma::load_matrix_sync(a_frag, a_ptr, A_STRIDE);

#pragma unroll
      for (int j = 0; j < FRAGS_N_PER_WARP; ++j) {
        const int col_offset = (warp_col_group * FRAGS_N_PER_WARP + j) * WMMA_N;
        const float *b_ptr = &Bs[kk * B_STRIDE + col_offset];

        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       wmma::precision::tf32, wmma::row_major>
            b_frag;
        wmma::load_matrix_sync(b_frag, b_ptr, B_STRIDE);
        wmma::mma_sync(c_frag[j], a_frag, b_frag, c_frag[j]);
      }
    }

    __syncthreads();
  }

  const int c_row = row_base + warp_row * WMMA_M;
#pragma unroll
  for (int j = 0; j < FRAGS_N_PER_WARP; ++j) {
    const int c_col =
        col_base + (warp_col_group * FRAGS_N_PER_WARP + j) * WMMA_N;
    wmma::store_matrix_sync(C + c_row * N + c_col, c_frag[j], N,
                            wmma::mem_row_major);
  }
}

static int get_benchmark_n(int argc, char *argv[]) {
  if (argc < 3) {
    return 4096;
  }
  const char *value = argv[2];
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0) {
    fprintf(stderr, "Invalid benchmark size: %s\n", value);
    exit(EXIT_FAILURE);
  }
  return static_cast<int>(parsed);
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
    size_t shared_bytes = (BLOCK_M * A_STRIDE + BLOCK_K * B_STRIDE) * sizeof(float);

    matmul_kernel_tensor_core_multistage<<<grid, block, shared_bytes>>>(A_d, B_d,
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
      const int N = get_benchmark_n(argc, argv);
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
