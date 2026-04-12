#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

/*
 Q, K, V, O: BxNhxNxD
*/
__global__ void flash_attention(
    float* Q, float* K, float* V, float* O,
    float* L, float* M,
    int B, int Nh, int N, int D,
    int Br, int Bc
) {
    const float scale = 1.0 / sqrtf(static_cast<float>(D));

    int Tr = (N+Br-1)/Br;
    int Tc = (N+Bc-1)/Bc;
    
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tx = threadIdx.x;
    
    int qkv_base = (b * (Nh*N*D)) + (h * (N*D));
    int lm_base = (b * (Nh*N)) + (h * N);

    extern __shared__ float smem[];
    float* Qi = smem; // Size = Br x d
    float* Kj = Qi + (Br * D); // Size = Bc x D
    float* Vj = Kj + (Bc * D); // Size = Bc x D
    float* S = Vj + (Bc * D); // Size = Br x Bc

    // outer loop: over K/V
    for (int j = 0; j < Tc; j++) {
        // load Kj and Vj
        for (int x = 0; x < D; x++) {
            Kj[tx*D + x] = K[qkv_base + (j*Bc + tx)*D + x];
            Vj[tx*D + x] = V[qkv_base + (j*Bc + tx)*D + x];
        }
        
        __syncthreads();
        
        // inner loop: over Q
        for (int i = 0; i < Tr; i++) {
            // load Qi
            for (int x = 0; x < D; x++) {
                Qi[tx*D + x] = Q[qkv_base + (i*Br + tx)*D + x];
            }

            // load l and m
            float l_prev = L[lm_base + i*Br + tx];
            float m_prev = M[lm_base + i*Br + tx];

            // calculate S_tx
            float max_score_tx = -INFINITY;
            for (int k = 0; k < Bc; k++) {
                float score = 0;
                for (int x = 0; x < D; x++) {
                    score += Qi[tx*D + x] * Kj[k*D + x];
                }
                score = score * scale;
                if (score > max_score_tx) max_score_tx = score;
                S[tx*Bc + k] = score;
            }

            float m_new = fmaxf(m_prev, max_score_tx);

            // normalize and sum S_tx
            float sum_score_tx = 0;
            for (int k = 0; k < Bc; k++) {
                float res = expf(S[tx*Bc + k] - m_new);
                sum_score_tx += res;
                S[tx*Bc + k] = res;
            }

            float exp_diff = expf(m_prev - m_new);
            float l_new = l_prev * exp_diff + sum_score_tx;

            // update M and L
            M[lm_base + i*Br + tx] = m_new;
            L[lm_base + i*Br + tx] = l_new;
            
            // update Oi
            for (int x = 0; x < D; x++) {
                float sv_dot = 0.0f;
                for (int k = 0; k < Bc; k++) {
                    sv_dot += S[tx*Bc + k] * Vj[k*D + x];
                }
                
                int out_idx = qkv_base + (i*Br + tx)*D + x;
                O[out_idx] = (O[out_idx] * l_prev * exp_diff + sv_dot) / l_new;
            }
        }

        __syncthreads();
    }
}

// Expects row-major tensors of shape [B * Nh * N * D]
void cpu_attention_batched(const float* Q, const float* K, const float* V, 
                           float* Out, int B, int Nh, int N, int D) {
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < Nh; ++h) {
            // Offset to the start of the current head
            int head_offset = (b * Nh + h) * (N * D);

            for (int i = 0; i < N; ++i) {
                std::vector<float> scores(N);
                float row_max = -INFINITY;

                // Step A: Matmul (Q * K^T) and Scaling
                for (int j = 0; j < N; ++j) {
                    float dot = 0.0f;
                    for (int k = 0; k < D; ++k) {
                        dot += Q[head_offset + i * D + k] * K[head_offset + j * D + k];
                    }
                    scores[j] = dot * scale;
                    if (scores[j] > row_max) row_max = scores[j];
                }

                // Step B: Numerically Stable Softmax
                float exp_sum = 0.0f;
                for (int j = 0; j < N; ++j) {
                    scores[j] = std::exp(scores[j] - row_max);
                    exp_sum += scores[j];
                }
                for (int j = 0; j < N; ++j) {
                    scores[j] /= exp_sum;
                }

                // Step C: Matmul (Softmax_probs * V)
                for (int j = 0; j < D; ++j) {
                    float out_val = 0.0f;
                    for (int k = 0; k < N; ++k) {
                        out_val += scores[k] * V[head_offset + k * D + j];
                    }
                    Out[head_offset + i * D + j] = out_val;
                }
            }
        }
    }
}

bool verify_results(const float* cpu_res, const float* gpu_res, int size) {
    float epsilon = 1e-4f;
    for (int i = 0; i < size; ++i) {
        if (std::abs(cpu_res[i] - gpu_res[i]) > epsilon) {
            std::cout << "Mismatch at " << i << ": CPU=" << cpu_res[i] << " GPU=" << gpu_res[i] << std::endl;
            return false;
        }
    }
    return true;
}

#define B 2
#define Nh 8
#define N 64
#define D 64
#define Br 32
#define Bc 32

int main() {
    const size_t total_elements = B * Nh * N * D;
    const size_t total_size = total_elements * sizeof(float);

    float* hQ = (float*)malloc(total_size);
    float* hK = (float*)malloc(total_size);
    float* hV = (float*)malloc(total_size);
    float* hO_CPU = (float*)malloc(total_size);
    float* hO_GPU = (float*)malloc(total_size);
    
    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, total_size);
    cudaMalloc(&dK, total_size);
    cudaMalloc(&dV, total_size);
    cudaMalloc(&dO, total_size);
    
    const size_t total_stats = B*Nh*N;
    const size_t total_stats_size = total_stats * sizeof(float);
    
    float *hM, *dM, *hL, *dL;
    hM = (float*)malloc(total_stats_size); cudaMalloc(&dM, total_stats_size);
    hL = (float*)malloc(total_stats_size); cudaMalloc(&dL, total_stats_size);

    // Initialize data
    for (int i = 0; i < total_elements; ++i) {
        hQ[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        hK[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        hV[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    for (int i = 0; i < total_stats; i++) {
        hM[i] = -INFINITY;
        hL[i] = 0;
    }

    // Copy data
    cudaMemcpy(dQ, hQ, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dM, hM, total_stats_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dL, hL, total_stats_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Run kernel
    dim3 threadsPerBlock(Br);
    dim3 blocksPerGrid(B, Nh);
    size_t shared_mem_size = ((Br*D) + (2*Bc*D) + (Br*Bc)) * sizeof(float);
    cudaEventRecord(start);
    flash_attention<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(
        dQ, dK, dV, dO,
        dL, dM,
        B, Nh, N, D,
        Br, Bc
    );
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Kernel Time: " << milliseconds << "ms\n";
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    // Copy results back
    cudaMemcpy(hO_GPU, dO, total_size, cudaMemcpyDeviceToHost);

    // Verify results
    std::cout << "Running CPU reference..." << std::endl;
    cpu_attention_batched(hQ, hK, hV, hO_CPU, B, Nh, N, D);
    if (verify_results(hO_CPU, hO_GPU, total_elements)) {
        std::cout << "SUCCESS: GPU results match CPU reference." << std::endl;
    } else {
        std::cout << "FAILURE: Results do not match." << std::endl;
    }

    // Cleanup
    free(hQ); free(hK); free(hV); free(hO_CPU); free(hO_GPU);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    
    return 0;
}

