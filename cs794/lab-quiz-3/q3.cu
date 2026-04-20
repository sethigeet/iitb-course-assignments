#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

// ======================
// START OF STUDENT CODE
// ======================

__global__ void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// matmul_AT: computes C = A^T * B
__global__ void matmul_AT(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[k * M + row] * B[k * N + col];  // A^T[row,k] = A[k,row]
        C[row * N + col] = sum;
    }
}

// matmul_BT: computes C = A * B^T
__global__ void matmul_BT(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[col * K + k];  // B^T[k,col] = B[col,k]
        C[row * N + col] = sum;
    }
}

__global__ void add_bias(float* Z, const float* b, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
        Z[row * N + col] += b[col];
}

__global__ void relu(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) x[idx] = fmaxf(0.0f, x[idx]);
}

/**
 * @brief Kernel to compute the derivative of ReLU activation for backpropagation. Check on how it is called to decide what each thread should compute.
 *
 * @param x Input array containing pre-activation values (Z) from the forward pass.
 * @param dx Input/output array containing the gradients flowing back from the upper layers. This array will be modified in-place to contain the gradients after applying the ReLU derivative.
 * @param size The total number of elements in the x and dx arrays.
 */ 
__global__ void relu_derivative(const float* x, float* dx, int size) {
    // TODO: implement ReLU derivative: dx[i] = (x[i] > 0) ? dx[i] : 0
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (x[idx] <= 0) dx[idx] = 0.0f;
    }
}

__global__ void mse_loss_grad(const float* pred, const float* target, float* dL, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) dL[idx] = (pred[idx] - target[idx]) * 2.0f / size;
}

/**
 * @brief Kernel to compute the bias gradients for a given layer during backpropagation. Check on how it is called to decide what each thread should compute.
 * @param dZ The input array containing the gradients with respect to the pre-activation outputs (Z) of the layer. This array has dimensions [N x D].
 * @param db The output array where the computed bias gradients will be stored. This array should have size D, corresponding to the number of neurons in the layer.
 * @param N The number of samples in the batch.
 * @param D The number of neurons in the layer (the dimension of the bias vector). 
 */
__global__ void bias_gradient(const float* dZ, float* db, int N, int D) {
    // TODO: compute bias gradient: db[j] = sum over i of dZ[i * D + j]
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < D) {
        float res = 0.0f;
        for (int i = 0; i < N; i++) {
            res += dZ[i*D + j];
        }
        db[j] = res;
    }
}

__global__ void sgd_update(float* W, const float* dW, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) W[idx] -= lr * dW[idx];
}

static void compute_A1(
    float* X, float* W1, float* b1,
    float* Z1_out, float* A1_out,
    int N, int IN, int H1)
{
    dim3 block(16, 16);
    int threads = 256;
    dim3 grid1((H1 + 15) / 16, (N + 15) / 16);
    matmul<<<grid1, block>>>(X, W1, Z1_out, N, IN, H1);
    cudaDeviceSynchronize();
    add_bias<<<grid1, block>>>(Z1_out, b1, N, H1);
    cudaMemcpy(A1_out, Z1_out, N * H1 * sizeof(float), cudaMemcpyDeviceToDevice);
    dim3 relu1_grid((N * H1 + threads - 1) / threads);
    dim3 relu1_block(threads);
    relu<<<relu1_grid, relu1_block>>>(A1_out, N * H1);
}

void forward_pass(
    float* X,
    float* W1, float* b1,
    float* W2, float* b2,
    float* W3, float* b3,
    float* A2,   // [out] stored activation after layer 2
    float* Z3,   // [out] stored pre-activation output of layer 3
    int N, int IN, int H1, int H2, int OUT)
{
    dim3 block(16, 16);
    int threads = 256;

    float *Z1_tmp, *A1_tmp;
    cudaMalloc(&Z1_tmp, N * H1 * sizeof(float));
    cudaMalloc(&A1_tmp, N * H1 * sizeof(float));

    // Layer 1  (checkpoint – computed but not kept)
    compute_A1(X, W1, b1, Z1_tmp, A1_tmp, N, IN, H1);
    cudaDeviceSynchronize(); 
    // Layer 2  (A2 is kept for backward)
    // Z2 = A1 · W2 + b2, then ReLU → A2
    // A1: [N x H1], W2: [H1 x H2], Z2/A2: [N x H2]
    dim3 grid2((H2 + 15) / 16, (N + 15) / 16);
    matmul<<<grid2, block>>>(A1_tmp, W2, A2, N, H1, H2);
    cudaDeviceSynchronize();
    add_bias<<<grid2, block>>>(A2, b2, N, H2);
    dim3 relu2_grid((N * H2 + threads - 1) / threads);
    dim3 relu2_block(threads);
    relu<<<relu2_grid, relu2_block>>>(A2, N * H2);

    // Layer 3  (output, Z3 is kept for backward)
    // Z3 = A2 · W3 + b3
    // A2: [N x H2], W3: [H2 x OUT], Z3: [N x OUT]
    dim3 grid3((OUT + 15) / 16, (N + 15) / 16);
    matmul<<<grid3, block>>>(A2, W3, Z3, N, H2, OUT);
    cudaDeviceSynchronize();
    add_bias<<<grid3, block>>>(Z3, b3, N, OUT);

    cudaFree(Z1_tmp);
    cudaFree(A1_tmp);
}

void backward_pass(
    float* X, float* target,
    float* W1, float* W2, float* W3,
    float* b1, float* b2, float* b3,
    float* A2,   // stored from forward pass
    float* Z3,   // stored from forward pass
    float* dW1, float* dW2, float* dW3,
    float* db1, float* db2, float* db3,
    float lr,
    int N, int IN, int H1, int H2, int OUT)
{
    int threads = 256;
    dim3 block(16, 16);
    dim3 elem_block(threads);

    // TODO: Recompute checkpointed layer 1 activations
    float *Z1, *A1;
    cudaMalloc(&Z1, N * H1 * sizeof(float));
    cudaMalloc(&A1, N * H1 * sizeof(float));
    compute_A1(X, W1, b1, Z1, A1, N, IN, H1);

    cudaDeviceSynchronize();
    
    // TODO: Recompute Z2 (pre-ReLU) — needed for relu_derivative
    // Z2 = A1 · W2 + b2  [N x H2]
    float *Z2;
    cudaMalloc(&Z2, N * H2 * sizeof(float));
    dim3 grid2((H2 + 15) / 16, (N + 15) / 16);
    matmul<<<grid2, block>>>(A1, W2, Z2, N, H1, H2);
    add_bias<<<grid2, block>>>(Z2, b2, N, H2);

    cudaDeviceSynchronize();

    // Gradient buffers
    float *dZ1, *dZ2, *dZ3;
    cudaMalloc(&dZ1, N * H1 * sizeof(float));
    cudaMalloc(&dZ2, N * H2 * sizeof(float));
    cudaMalloc(&dZ3, N * OUT * sizeof(float));

    // TODO: Layer 3 backward (uses stored Z3 and A2)
    // dZ3 = MSE gradient  [N x OUT]
    // dW3 = A2^T · dZ3   [H2 x OUT]
    // A2 is [N x H2], so A2^T is [H2 x N]; dZ3 is [N x OUT]
    // Use matmul_AT: A stored as [K x M] = [N x H2], M=H2, K=N, N=OUT
    mse_loss_grad<<<dim3(((N*OUT) + 255) / 256), elem_block>>>(Z3, target, dZ3, N*OUT);
    matmul_AT<<<dim3((N + 15) / 16, (H2 + 15) / 16), block>>>(A2, dZ3, dW3, H2, N, OUT);
    cudaDeviceSynchronize();
    bias_gradient<<<dim3((OUT + 255) / 256), elem_block>>>(dZ3, db3, N, OUT);

    // Layer 2 backward (uses recomputed Z2 and stored A2)
    // dZ2 = dZ3 · W3^T   [N x H2]
    // dZ3 is [N x OUT], W3 is [H2 x OUT], so W3^T is [OUT x H2]
    // Use matmul_BT: A=[N x OUT], B stored as [N_out x K]=[H2 x OUT], M=N, K=OUT, N=H2
    matmul_BT<<<dim3((H2 + 15) / 16, (N + 15) / 16), block>>>(dZ3, W3, dZ2, N, OUT, H2);
    cudaDeviceSynchronize();
    relu_derivative<<<dim3((N * H2 + 255) / 256), elem_block>>>(Z2, dZ2, N * H2);
    //  apply ReLU derivative to dZ2 using Z2

    // dW2 = A1^T · dZ2   [H1 x H2]
    // A1 is [N x H1], so A1^T is [H1 x N]; dZ2 is [N x H2]
    // Use matmul_AT: A stored as [K x M]=[N x H1], M=H1, K=N, N=H2
    matmul_AT<<<dim3((H2 + 15) / 16, (H1 + 15) / 16), block>>>(A1, dZ2, dW2, H1, N, H2);
    cudaDeviceSynchronize();
    bias_gradient<<<dim3((H2 + 255) / 256), elem_block>>>(dZ2, db2, N, H2);

    // Layer 1 backward  (uses recomputed Z1) 
    // dZ1 = dZ2 · W2^T   [N x H1]
    // dZ2 is [N x H2], W2 is [H1 x H2], so W2^T is [H2 x H1]
    // Use matmul_BT: A=[N x H2], B stored as [N_out x K]=[H1 x H2], M=N, K=H2, N=H1
    matmul_BT<<<dim3((H1 + 15) / 16, (N + 15) / 16), block>>>(dZ2, W2, dZ1, N, H2, H1);
    cudaDeviceSynchronize();
    relu_derivative<<<dim3((N * H1 + 255) / 256), elem_block>>>(Z1, dZ1, N * H1);
    // dW1 = X^T · dZ1    [IN x H1]
    // X is [N x IN], so X^T is [IN x N]; dZ1 is [N x H1]
    // Use matmul_AT: A stored as [K x M]=[N x IN], M=IN, K=N, N=H1
    matmul_AT<<<dim3((H1 + 15) / 16, (IN + 15) / 16), block>>>(X, dZ1, dW1, IN, N, H1);
    cudaDeviceSynchronize();
    bias_gradient<<<dim3((H1 + 255) / 256), elem_block>>>(dZ1, db1, N, H1);
    cudaDeviceSynchronize();
    
    // TODO: SGD update for W1, b1, W2, b2, W3, b3 (6 calls)
    sgd_update<<<dim3(((IN*H1) + 255) / 256), elem_block>>>(W1, dW1, lr, IN*H1);
    sgd_update<<<dim3(((H1) + 255) / 256), elem_block>>>(b1, db1, lr, H1);
    sgd_update<<<dim3(((H1*H2) + 255) / 256), elem_block>>>(W2, dW2, lr, H1*H2);
    sgd_update<<<dim3(((H2) + 255) / 256), elem_block>>>(b2, db2, lr, H2);
    sgd_update<<<dim3(((H2*OUT) + 255) / 256), elem_block>>>(W3, dW3, lr, H2*OUT);
    sgd_update<<<dim3(((OUT) + 255) / 256), elem_block>>>(b3, db3, lr, OUT);
    cudaDeviceSynchronize();

    cudaFree(Z1);  cudaFree(A1);
    cudaFree(Z2);
    cudaFree(dZ1); cudaFree(dZ2); cudaFree(dZ3);
}


// ====================
// END OF STUDENT CODE
// ====================

// ======================================================================
// TESTING INFRASTRUCTURE: PLEASE DON'T CHANGE ANYTHING BELOW THIS LINE.
// ======================================================================

// ============================================================
// CPU REFERENCE IMPLEMENTATION
// ============================================================

static void cpu_relu(float* x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

static void cpu_matmul(const float* A, const float* B, float* C,
                       int M, int K, int N)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

static void cpu_add_bias(float* C, const float* bias, int M, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i * N + j] += bias[j];
}

static void cpu_transpose(const float* A, float* AT, int R, int C) {
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            AT[j * R + i] = A[i * C + j];
}

static void cpu_forward(
    const float* X,
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    const float* W3, const float* b3,
    float* Z1, float* A1,
    float* Z2, float* A2,
    float* Z3,
    int N, int IN, int H1, int H2, int OUT)
{
    // Layer 1
    cpu_matmul(X,  W1, Z1, N, IN, H1);
    cpu_add_bias(Z1, b1, N, H1);
    memcpy(A1, Z1, N * H1 * sizeof(float));
    cpu_relu(A1, N * H1);

    // Layer 2
    cpu_matmul(A1, W2, Z2, N, H1, H2);
    cpu_add_bias(Z2, b2, N, H2);
    memcpy(A2, Z2, N * H2 * sizeof(float));
    cpu_relu(A2, N * H2);

    // Layer 3
    cpu_matmul(A2, W3, Z3, N, H2, OUT);
    cpu_add_bias(Z3, b3, N, OUT);
}

static void cpu_backward(
    const float* X, const float* target,
    float* W1, float* b1,
    float* W2, float* b2,
    float* W3, float* b3,
    const float* Z1, const float* A1,
    const float* Z2, const float* A2,
    const float* Z3,
    float* dW1, float* db1,
    float* dW2, float* db2,
    float* dW3, float* db3,
    float lr,
    int N, int IN, int H1, int H2, int OUT)
{
    // dL/dZ3 = MSE gradient  [N x OUT]
    float* dZ3 = (float*)malloc(N * OUT * sizeof(float));
    int tot3 = N * OUT;
    for (int i = 0; i < tot3; i++)
        dZ3[i] = (Z3[i] - target[i]) * 2.0f / tot3;

    // Layer 3 weight gradients
    float* A2T = (float*)malloc(H2 * N * sizeof(float));
    cpu_transpose(A2, A2T, N, H2);
    cpu_matmul(A2T, dZ3, dW3, H2, N, OUT);
    free(A2T);

    for (int j = 0; j < OUT; j++) {
        float s = 0.0f;
        for (int i = 0; i < N; i++) s += dZ3[i * OUT + j];
        db3[j] = s;
    }

    // Layer 2 backward
    float* dZ2 = (float*)malloc(N * H2 * sizeof(float));
    float* W3T = (float*)malloc(OUT * H2 * sizeof(float));
    cpu_transpose(W3, W3T, H2, OUT);
    cpu_matmul(dZ3, W3T, dZ2, N, OUT, H2);
    free(W3T);

    for (int i = 0; i < N * H2; i++)
        dZ2[i] *= (Z2[i] > 0.0f) ? 1.0f : 0.0f;

    float* A1T = (float*)malloc(H1 * N * sizeof(float));
    cpu_transpose(A1, A1T, N, H1);
    cpu_matmul(A1T, dZ2, dW2, H1, N, H2);
    free(A1T);

    for (int j = 0; j < H2; j++) {
        float s = 0.0f;
        for (int i = 0; i < N; i++) s += dZ2[i * H2 + j];
        db2[j] = s;
    }

    // Layer 1 backward
    float* dZ1 = (float*)malloc(N * H1 * sizeof(float));
    float* W2T = (float*)malloc(H2 * H1 * sizeof(float));
    cpu_transpose(W2, W2T, H1, H2);
    cpu_matmul(dZ2, W2T, dZ1, N, H2, H1);
    free(W2T);

    for (int i = 0; i < N * H1; i++)
        dZ1[i] *= (Z1[i] > 0.0f) ? 1.0f : 0.0f;

    float* XT = (float*)malloc(IN * N * sizeof(float));
    cpu_transpose(X, XT, N, IN);
    cpu_matmul(XT, dZ1, dW1, IN, N, H1);
    free(XT);

    for (int j = 0; j < H1; j++) {
        float s = 0.0f;
        for (int i = 0; i < N; i++) s += dZ1[i * H1 + j];
        db1[j] = s;
    }

    // SGD updates
    for (int i = 0; i < IN * H1; i++) W1[i] -= lr * dW1[i];
    for (int i = 0; i < H1;      i++) b1[i] -= lr * db1[i];
    for (int i = 0; i < H1 * H2; i++) W2[i] -= lr * dW2[i];
    for (int i = 0; i < H2;      i++) b2[i] -= lr * db2[i];
    for (int i = 0; i < H2 * OUT;i++) W3[i] -= lr * dW3[i];
    for (int i = 0; i < OUT;     i++) b3[i] -= lr * db3[i];

    free(dZ3); free(dZ2); free(dZ1);
}

// ============================================================
// TESTING INFRASTRUCTURE
// ============================================================

static void fill_random(float* data, int n, unsigned int* seed) {
    for (int i = 0; i < n; i++)
        data[i] = ((float)(rand_r(seed) % 2001) - 1000.0f) / 1000.0f;
}

static int compare_arrays(const char* name,
                           const float* ref, const float* got,
                           int n, float tol)
{
    int ok = 1;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - got[i]);
        if (diff > tol) {
            printf("  [%s] MISMATCH at idx %d: cpu=%.6f  gpu=%.6f  diff=%.2e\n",
                   name, i, ref[i], got[i], diff);
            ok = 0;
            break;
        }
    }
    return ok;
}

static void gpu_forward_backward(
    const float* h_X, const float* h_target,
    float* h_W1, float* h_b1,
    float* h_W2, float* h_b2,
    float* h_W3, float* h_b3,
    float* h_dW1, float* h_db1,
    float* h_dW2, float* h_db2,
    float* h_dW3, float* h_db3,
    float* h_Z3_out,
    float lr,
    int N, int IN, int H1, int H2, int OUT)
{
    float *X, *target;
    float *W1, *b1, *W2, *b2, *W3, *b3;
    float *A2, *Z3;
    float *dW1, *db1, *dW2, *db2, *dW3, *db3;

    cudaMalloc(&X,      N * IN  * sizeof(float));
    cudaMalloc(&target, N * OUT * sizeof(float));
    cudaMalloc(&W1, IN * H1 * sizeof(float));  cudaMalloc(&b1, H1  * sizeof(float));
    cudaMalloc(&W2, H1 * H2 * sizeof(float));  cudaMalloc(&b2, H2  * sizeof(float));
    cudaMalloc(&W3, H2 * OUT* sizeof(float));  cudaMalloc(&b3, OUT * sizeof(float));
    cudaMalloc(&A2, N  * H2  * sizeof(float));
    cudaMalloc(&Z3, N  * OUT * sizeof(float));
    cudaMalloc(&dW1, IN * H1 * sizeof(float)); cudaMemset(dW1, 0, IN * H1 * sizeof(float));
    cudaMalloc(&db1, H1  * sizeof(float));     cudaMemset(db1, 0, H1  * sizeof(float));
    cudaMalloc(&dW2, H1 * H2 * sizeof(float)); cudaMemset(dW2, 0, H1 * H2 * sizeof(float));
    cudaMalloc(&db2, H2  * sizeof(float));     cudaMemset(db2, 0, H2  * sizeof(float));
    cudaMalloc(&dW3, H2 * OUT* sizeof(float)); cudaMemset(dW3, 0, H2 * OUT* sizeof(float));
    cudaMalloc(&db3, OUT * sizeof(float));     cudaMemset(db3, 0, OUT * sizeof(float));

    cudaMemcpy(X,      h_X,      N * IN  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(target, h_target, N * OUT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W1, h_W1, IN * H1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b1, h_b1, H1      * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W2, h_W2, H1 * H2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b2, h_b2, H2      * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W3, h_W3, H2 * OUT* sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b3, h_b3, OUT     * sizeof(float), cudaMemcpyHostToDevice);

    forward_pass(X, W1, b1, W2, b2, W3, b3, A2, Z3, N, IN, H1, H2, OUT);
    cudaDeviceSynchronize();

    backward_pass(X, target,
                  W1, W2, W3, b1, b2, b3,
                  A2, Z3,
                  dW1, dW2, dW3, db1, db2, db3,
                  lr, N, IN, H1, H2, OUT);
    cudaDeviceSynchronize();

    cudaMemcpy(h_W1, W1, IN * H1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b1, b1, H1      * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_W2, W2, H1 * H2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b2, b2, H2      * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_W3, W3, H2 * OUT* sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b3, b3, OUT     * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dW1, dW1, IN * H1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_db1, db1, H1      * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dW2, dW2, H1 * H2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_db2, db2, H2      * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dW3, dW3, H2 * OUT* sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_db3, db3, OUT     * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Z3_out, Z3, N * OUT * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(X);  cudaFree(target);
    cudaFree(W1); cudaFree(b1); cudaFree(W2); cudaFree(b2);
    cudaFree(W3); cudaFree(b3);
    cudaFree(A2); cudaFree(Z3);
    cudaFree(dW1); cudaFree(db1); cudaFree(dW2); cudaFree(db2);
    cudaFree(dW3); cudaFree(db3);
}

static void run_test(const char* label,
                     int N, int IN, int H1, int H2, int OUT,
                     float lr, float tol)
{
    printf("=== %s: N=%d IN=%d H1=%d H2=%d OUT=%d lr=%.4f ===\n",
           label, N, IN, H1, H2, OUT, lr);

    unsigned int seed = 42;

    float *h_X      = (float*)malloc(N * IN  * sizeof(float));
    float *h_target = (float*)malloc(N * OUT * sizeof(float));

    float *gW1 = (float*)malloc(IN * H1 * sizeof(float));
    float *gb1 = (float*)malloc(H1       * sizeof(float));
    float *gW2 = (float*)malloc(H1 * H2 * sizeof(float));
    float *gb2 = (float*)malloc(H2       * sizeof(float));
    float *gW3 = (float*)malloc(H2 * OUT * sizeof(float));
    float *gb3 = (float*)malloc(OUT      * sizeof(float));

    float *cW1 = (float*)malloc(IN * H1 * sizeof(float));
    float *cb1 = (float*)malloc(H1       * sizeof(float));
    float *cW2 = (float*)malloc(H1 * H2 * sizeof(float));
    float *cb2 = (float*)malloc(H2       * sizeof(float));
    float *cW3 = (float*)malloc(H2 * OUT * sizeof(float));
    float *cb3 = (float*)malloc(OUT      * sizeof(float));

    fill_random(h_X,      N * IN,  &seed);
    fill_random(h_target, N * OUT, &seed);
    fill_random(gW1, IN * H1, &seed);  fill_random(gb1, H1,  &seed);
    fill_random(gW2, H1 * H2, &seed);  fill_random(gb2, H2,  &seed);
    fill_random(gW3, H2 * OUT,&seed);  fill_random(gb3, OUT, &seed);

    memcpy(cW1, gW1, IN * H1 * sizeof(float));
    memcpy(cb1, gb1, H1       * sizeof(float));
    memcpy(cW2, gW2, H1 * H2 * sizeof(float));
    memcpy(cb2, gb2, H2       * sizeof(float));
    memcpy(cW3, gW3, H2 * OUT * sizeof(float));
    memcpy(cb3, gb3, OUT      * sizeof(float));

    float *gdW1 = (float*)calloc(IN * H1, sizeof(float));
    float *gdb1 = (float*)calloc(H1,       sizeof(float));
    float *gdW2 = (float*)calloc(H1 * H2, sizeof(float));
    float *gdb2 = (float*)calloc(H2,       sizeof(float));
    float *gdW3 = (float*)calloc(H2 * OUT, sizeof(float));
    float *gdb3 = (float*)calloc(OUT,      sizeof(float));
    float *gZ3  = (float*)calloc(N * OUT,  sizeof(float));

    float *cdW1 = (float*)calloc(IN * H1, sizeof(float));
    float *cdb1 = (float*)calloc(H1,       sizeof(float));
    float *cdW2 = (float*)calloc(H1 * H2, sizeof(float));
    float *cdb2 = (float*)calloc(H2,       sizeof(float));
    float *cdW3 = (float*)calloc(H2 * OUT, sizeof(float));
    float *cdb3 = (float*)calloc(OUT,      sizeof(float));

    float *cZ1 = (float*)malloc(N * H1  * sizeof(float));
    float *cA1 = (float*)malloc(N * H1  * sizeof(float));
    float *cZ2 = (float*)malloc(N * H2  * sizeof(float));
    float *cA2 = (float*)malloc(N * H2  * sizeof(float));
    float *cZ3 = (float*)malloc(N * OUT * sizeof(float));

    cpu_forward(h_X, cW1, cb1, cW2, cb2, cW3, cb3,
                cZ1, cA1, cZ2, cA2, cZ3,
                N, IN, H1, H2, OUT);

    cpu_backward(h_X, h_target,
                 cW1, cb1, cW2, cb2, cW3, cb3,
                 cZ1, cA1, cZ2, cA2, cZ3,
                 cdW1, cdb1, cdW2, cdb2, cdW3, cdb3,
                 lr, N, IN, H1, H2, OUT);

    gpu_forward_backward(h_X, h_target,
                         gW1, gb1, gW2, gb2, gW3, gb3,
                         gdW1, gdb1, gdW2, gdb2, gdW3, gdb3,
                         gZ3, lr,
                         N, IN, H1, H2, OUT);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error: %s\n  Result: FAIL\n\n", cudaGetErrorString(err));
        goto cleanup;
    }

    {
        int all_ok = 1;
        all_ok &= compare_arrays("Z3 (forward output)", cZ3,  gZ3,  N * OUT, tol);
        all_ok &= compare_arrays("dW1", cdW1, gdW1, IN * H1, tol);
        all_ok &= compare_arrays("db1", cdb1, gdb1, H1,       tol);
        all_ok &= compare_arrays("dW2", cdW2, gdW2, H1 * H2,  tol);
        all_ok &= compare_arrays("db2", cdb2, gdb2, H2,       tol);
        all_ok &= compare_arrays("dW3", cdW3, gdW3, H2 * OUT, tol);
        all_ok &= compare_arrays("db3", cdb3, gdb3, OUT,      tol);
        all_ok &= compare_arrays("W1 (post-update)", cW1, gW1, IN * H1, tol);
        all_ok &= compare_arrays("b1 (post-update)", cb1, gb1, H1,       tol);
        all_ok &= compare_arrays("W2 (post-update)", cW2, gW2, H1 * H2,  tol);
        all_ok &= compare_arrays("b2 (post-update)", cb2, gb2, H2,       tol);
        all_ok &= compare_arrays("W3 (post-update)", cW3, gW3, H2 * OUT, tol);
        all_ok &= compare_arrays("b3 (post-update)", cb3, gb3, OUT,      tol);
        printf("  Result: %s\n\n", all_ok ? "PASS" : "FAIL");
    }

cleanup:
    free(h_X); free(h_target);
    free(gW1); free(gb1); free(gW2); free(gb2); free(gW3); free(gb3);
    free(cW1); free(cb1); free(cW2); free(cb2); free(cW3); free(cb3);
    free(gdW1); free(gdb1); free(gdW2); free(gdb2); free(gdW3); free(gdb3); free(gZ3);
    free(cdW1); free(cdb1); free(cdW2); free(cdb2); free(cdW3); free(cdb3);
    free(cZ1); free(cA1); free(cZ2); free(cA2); free(cZ3);
}

// ============================================================
// MAIN
// ============================================================

int main(void)
{
    srand(42);

    run_test("Test1", /*N=*/4, /*IN=*/8, /*H1=*/8, /*H2=*/8, /*OUT=*/4, /*lr=*/0.01f, /*tol=*/1e-3f);

    run_test("Test2", /*N=*/16, /*IN=*/32, /*H1=*/32, /*H2=*/32, /*OUT=*/16, /*lr=*/0.001f, /*tol=*/1e-3f);

    run_test("Test3", /*N=*/32, /*IN=*/64, /*H1=*/128, /*H2=*/64, /*OUT=*/32, /*lr=*/0.001f, /*tol=*/1e-3f);

    return 0;
}

