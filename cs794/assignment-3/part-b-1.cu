#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

/*
 A: MxK
 B: KxN
 C: MxN
*/
__global__ void matmul(float* A, float* B, float* C, int M, int K, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    float res = 0;
    for (int i = 0; i < K; i++) {
        if (y < M && x < N)
            res += A[y*K+i] * B[i*N+x];
    }
    
    if (y < M && x < N)
        C[y*N+x] = res;
}

/*
 A: MxK
 B: KxN
 C: MxN
 NOTE: these dimensions should be the ones after the transpose and not before
*/
__global__ void transposed_matmul(float* A, float* B, float* C, int M, int K, int N, bool transposeA, bool transposeB) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    float res = 0;
    for (int i = 0; i < K; i++) {
        if (y < M && x < N) {
            float a;
            if (transposeA) a = A[i*M+y]; // input A is KxM
            else a = A[y*K+i]; // input A is MxK
            
            float b;
            if (transposeB) b = B[x*K+i]; // input B is NxK
            else b = B[i*N+x]; // input B is KxN
            
            res += a * b;
        }
    }
    
    if (y < M && x < N)
        C[y*N+x] = res;
}

/*
 A: MxN
 b: N
 C: MxN
*/
__global__ void broadcast_add(float* A, float* b, float* C, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < M) {
        C[y*N + x] = A[y*N + x] + b[x];
    }
}

/*
 A: MxN
 B: MxN
 NOTE: A is also the destination
*/
__global__ void weighted_sum_inplace(float* A, float* B, int M, int N, float a, float b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < M) {
        A[y*N + x] = a * A[y*N + x] + b * B[y*N + x];
    }
}
/*
 A: N
 B: N
 NOTE: A is also the destination
*/
__global__ void weighted_sum_inplace(float* A, float* B, int N, float a, float b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < N) {
        A[x] = a * A[x] + b * B[x];
    }
}

/*
 A: MxN
 B: M
*/
__global__ void col_sum(float* A, float* B, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < N) {
        float res = 0;
        for (int i = 0; i < M; i++) {
            res += A[i*N+x];
        }
        B[x] = res;
    }
}

/*
 X: MxN
 A: MxN
*/
__global__ void relu(float* X, float* A, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < M) {
        if (X[y*N + x] < 0) A[y*N + x] = 0;
        else A[y*N + x] = X[y*N + x];
    }
}

/*
 src: MxN
 reference: MxN
 dst: MxN
*/
__global__ void relu_gradient(float* src, float* reference, float* dst, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < M) {
        if (reference[y*N + x] < 0) dst[y*N + x] = 0;
        else dst[y*N + x] = src[y*N + x];
    }
}

/*
 actual: MxN
 reference: MxN
 dst: MxN
*/
__global__ void loss_gradient(float* actual, float* reference, float* dst, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < M) {
        dst[y*N+x] = (actual[y*N+x] - reference[y*N+x]) * (2.0/(M*N));
    }
}

float loss(float* actual, float* reference, int M, int N) {
    float res = 0.0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float diff = actual[i*N+j] - reference[i*N+j];
            res += diff * diff;
        }
    }

    return res / (M*N);
}

#define VSIZE(n) ((n) * sizeof(float))
#define MSIZE(m, n) ((m) * (n) * sizeof(float))

#define SEED 42

void randomize(float* ptr, size_t size, float limit) {
    static std::mt19937 gen(SEED);
    std::uniform_real_distribution<float> dis(-limit, limit);
    for (size_t i = 0; i < size; ++i) {
        ptr[i] = dis(gen);
    }
}

struct Layer {
    int in_dim, out_dim, batch_dim;
    
    // Parameters (Host + Device)
    float *hW, *dW;
    float *hb, *db;

    // Gradients & Activations (Device only)
    // Z: pre-activation, A: post-activation
    float *ddW, *ddb, *dZ, *ddZ, *dA, *ddA;

    Layer(int _in_dim, int _out_dim, int _batch_dim) {
        in_dim = _in_dim; out_dim = _out_dim; batch_dim = _batch_dim;
        
        hW = (float*)malloc(MSIZE(in_dim, out_dim)); cudaMalloc(&dW, MSIZE(in_dim, out_dim));
        hb = (float*)malloc(VSIZE(out_dim)); cudaMalloc(&db, VSIZE(out_dim));
        cudaMalloc(&ddW, MSIZE(in_dim, out_dim)); cudaMalloc(&ddb, VSIZE(out_dim));
        cudaMalloc(&dZ, MSIZE(batch_dim, out_dim)); cudaMalloc(&ddZ, MSIZE(batch_dim, out_dim));
        cudaMalloc(&dA, MSIZE(batch_dim, out_dim)); cudaMalloc(&ddA, MSIZE(batch_dim, out_dim));
    }

    ~Layer() {
        free(hW); cudaFree(dW);
        free(hb); cudaFree(db);
        cudaFree(ddW); cudaFree(ddb);
        cudaFree(dZ); cudaFree(ddZ);
        cudaFree(dA); cudaFree(ddA);
    }
    
    void initializeMatrices() {
        // Xavier initialization limit: sqrt(6 / (in + out))
        float w_limit = sqrt(6.0f / (in_dim + out_dim));
        
        randomize(hW, in_dim * out_dim, w_limit);
        randomize(hb, out_dim, 0.01f); // Small constant for biases
    }

    void copyToDevice() {
        cudaMemcpy(dW, hW, MSIZE(in_dim, out_dim), cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb, VSIZE(out_dim), cudaMemcpyHostToDevice);
    }

    void forward(float* input, bool applyRelu) {
        dim3 threadsPerBlock(32, 32);
        dim3 blocksPerGrid((out_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (batch_dim + threadsPerBlock.y - 1) / threadsPerBlock.y);
                    
        matmul<<<blocksPerGrid, threadsPerBlock>>>(input, dW, dZ, batch_dim, in_dim, out_dim);
        broadcast_add<<<blocksPerGrid, threadsPerBlock>>>(dZ, db, dZ, batch_dim, out_dim);
        if (applyRelu)
            relu<<<blocksPerGrid, threadsPerBlock>>>(dZ, dA, batch_dim, out_dim);
        cudaDeviceSynchronize();
    }

    void backward(float* input, Layer* nextLayer, bool compute_ddZ) {
        if (nextLayer) {
            dim3 threadsPerBlock(32, 32);
            dim3 blocksPerGrid((nextLayer->in_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (nextLayer->batch_dim + threadsPerBlock.y - 1) / threadsPerBlock.y);
            transposed_matmul<<<blocksPerGrid, threadsPerBlock>>>(nextLayer->ddZ, nextLayer->dW, ddA, nextLayer->batch_dim, nextLayer->out_dim, nextLayer->in_dim, false, true);
        }
        
        if (compute_ddZ) {
            dim3 threadsPerBlock(32, 32);
            dim3 blocksPerGrid((out_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (batch_dim + threadsPerBlock.y - 1) / threadsPerBlock.y);
            relu_gradient<<<blocksPerGrid, threadsPerBlock>>>(ddA, dZ, ddZ, batch_dim, out_dim);
        }

        {
            dim3 threadsPerBlock(32, 32);
            dim3 blocksPerGrid((out_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (batch_dim + threadsPerBlock.y - 1) / threadsPerBlock.y);
            transposed_matmul<<<blocksPerGrid, threadsPerBlock>>>(input, ddZ, ddW, in_dim, batch_dim, out_dim, true, false);
        }

        {
            dim3 threadsPerBlock(32);
            dim3 blocksPerGrid((out_dim + threadsPerBlock.x - 1) / threadsPerBlock.x);
            col_sum<<<blocksPerGrid, threadsPerBlock>>>(ddZ, ddb, batch_dim, out_dim);
        }
    }

    void step(float lr) {
        {
            dim3 threadsPerBlock(32, 32);
            dim3 blocksPerGrid((out_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (in_dim + threadsPerBlock.y - 1) / threadsPerBlock.y);
            weighted_sum_inplace<<<blocksPerGrid, threadsPerBlock>>>(dW, ddW, in_dim, out_dim, 1.0, -1.0 * lr);
        }
        
        {
            dim3 threadsPerBlock(32);
            dim3 blocksPerGrid((out_dim + threadsPerBlock.x - 1) / threadsPerBlock.x);
            weighted_sum_inplace<<<blocksPerGrid, threadsPerBlock>>>(db, ddb, out_dim, 1.0, -1.0 * lr);
        }
    }
};

void print_matrix(float* mat, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            cout << mat[i*N + j] << " ";
        }
        cout << "\n";
    }
}

// GPU Params
//#define N 1024
//#define IN 512
//#define H1 2048
//#define H2 2048
//#define H3 2048
//#define OUT 512
//#define EPOCHS 100
//#define LR 0.001
// CPU Params
#define N 2
#define IN 16
#define H1 32
#define H2 32
#define H3 32
#define OUT 16
#define EPOCHS 10
#define LR 0.01

int main() {
    vector<int> dims = {IN, H1, H2, H3, OUT};
    vector<Layer> layers;
    layers.reserve(dims.size());
    for (int l = 0; l < dims.size()-1; l++) {
        layers.emplace_back(dims[l], dims[l+1], N);
    }

    float *hIn, *dIn, *hOut; // dOut is already allocated at layers[-1].dZ (we don't apply relu on the last layer)
    float *hRef, *dRef;
    hIn = (float*)malloc(MSIZE(N, IN)); cudaMalloc(&dIn, MSIZE(N, IN));
    hOut = (float*)malloc(MSIZE(N, OUT));
    hRef = (float*)malloc(MSIZE(N, OUT)); cudaMalloc(&dRef, MSIZE(N, OUT));

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    cout << "Total memory: " << totalMem/1e9 << "GB\n";
    cout << "Total memory free: " << freeMem/1e9 << "GB\n";
    cout << "Total memory used: " << (totalMem - freeMem)/1e9 << "GB\n";

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < IN; j++) {
            hIn[i*IN+j] = 1;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < OUT; j++) {
            hRef[i*OUT+j] = i+j;
        }
    }
    
    // Copy matrices to device
    cudaMemcpy(dRef, hRef, MSIZE(N, OUT), cudaMemcpyHostToDevice);
    cudaMemcpy(dIn, hIn, MSIZE(N, IN), cudaMemcpyHostToDevice);
    for (auto& layer : layers) {
        layer.initializeMatrices();
        layer.copyToDevice();
    }

    // Training loop
    for (int e = 0; e < EPOCHS; e++) {
        // Forward pass
        layers[0].forward(dIn, true);
        for (int l = 1; l < layers.size(); l++) {
            // We don't apply relu on the last layer
            layers[l].forward(layers[l-1].dA, l != layers.size()-1);
        }
        float* dOut = layers[layers.size()-1].dZ;

        // Backward pass
        dim3 threadsPerBlock(32, 32);
        dim3 blocksPerGrid((OUT + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
        loss_gradient<<<blocksPerGrid, threadsPerBlock>>>(dOut, dRef, layers[layers.size()-1].ddZ, N, OUT);
        for (int l = layers.size()-1; l >= 0; l--) {
            float* input;
            if (l == 0) input = dIn;
            else input = layers[l-1].dA;

            Layer* nextLayer = nullptr;
            if (l != layers.size()-1) nextLayer = &layers[l+1];
            
            layers[l].backward(input, nextLayer, l != layers.size()-1);
        }
        
        // Parameter update
        for (int l = layers.size()-1; l >= 0; l--) {
            layers[l].step(LR);
        }

        if (e % 10 == 0) {
            cudaMemcpy(hOut, layers[layers.size()-1].dZ, MSIZE(N, OUT), cudaMemcpyDeviceToHost);
            cout << "Loss: " << loss(hOut, hRef, N, OUT) << endl;
        }
    }

    // Copy result to host
    cudaMemcpy(hOut, layers[layers.size()-1].dZ, MSIZE(N, OUT), cudaMemcpyDeviceToHost);

    // Print the result
    cout << "Input:";
    print_matrix(hIn, N, IN);
    cout << "\nReference:";
    print_matrix(hRef, N, OUT);
    cout << "\nFinal Output:";
    print_matrix(hOut, N, OUT);

    // Cleanup
    free(hIn); cudaFree(dIn);
    free(hRef); cudaFree(dRef);
    free(hOut);
    
    return 0;
}

