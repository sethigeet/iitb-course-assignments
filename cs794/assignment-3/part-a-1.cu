#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void matmul(float* M1, float* M2, float* R, int N, int B, int k) {
    extern __shared__ float tiles[];
    float* tileM1 = tiles;
    float* tileM2 = tiles + k*k;
    
    // Global indices inside the result matrix
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;
    // Local indices inside the tile
    int localX = threadIdx.x;
    int localY = threadIdx.y;

    bool valid = globalX < N && globalY < B;

    if (valid) R[globalY*N + globalX] = 0;
    __syncthreads();

    // Iterate over row/col of tiles
    for (int i = 0; i < gridDim.x; i++) {
        int M1x = i*k + localX;
        int M1y = globalY;
        int M2x = globalX;
        int M2y = localY + i*k;
        
        // Copy elements of A and B into shared memory
        if (M1x < N && M1y < B) {
            tileM1[localY*k + localX] = M1[M1y*N + M1x];
        } else {
            tileM1[localY*k + localX] = 0;
        }
        
        if (M2x < N && M2y < N) {
            tileM2[localY*k + localX] = M2[M2y * N + M2x];
        } else {
            tileM2[localY*k + localX] = 0;
        }

        __syncthreads();

        // Compute matmul
        if (valid) {
            float res = 0;
            for (int j = 0; j < k; j++) {
                res += tileM1[localY*k + j] * tileM2[j*k + localX];
            }
            R[globalY*N + globalX] += res;
        }

        __syncthreads();
    }
}

__global__ void relu(float* M, int N, int B) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (M[y*N + x] <= 0) M[y*N + x] = 0;
}

#define N 8
#define k 5
#define B 4

int main() {
    float *hW1, *hW2, *hX;
    float *dW1, *dW2, *dX, *dZ;

    // Allocate host and device memory
    cudaMallocHost(&hW1, N * N * sizeof(float));
    cudaMallocHost(&hW2, N * N * sizeof(float));
    cudaMallocHost(&hX, B * N * sizeof(float));
    cudaMalloc(&dW1, N * N * sizeof(float));
    cudaMalloc(&dW2, N * N * sizeof(float));
    cudaMalloc(&dX, B * N * sizeof(float));
    cudaMalloc(&dZ, B * N * sizeof(float));

    // Fill matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            hW1[i*N + j] = i-j;
            //if (i == j) {
            //    hW1[i*N + j] = 1;
            //}
            
            hW2[i*N + j] = 0;
            if (i == j) {
                hW2[i*N + j] = 1;
            }
            
            if (i < B) {
                hX[i*N + j] = i+j;
            }
        }
    }
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << hW1[i*N + j] << " ";
        }
        cout << "\n";
    }

    cout << "\n\n\n";
    
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < N; j++) {
            cout << hX[i*N + j] << " ";
        }
        cout << "\n";
    }
    cout << "\n\n\n";

    cudaMemcpy(dW1, hW1, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dW2, hW2, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, hX, B*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(k, k);
    dim3 gridSize((N+k-1)/k, (B+k-1)/k);
    matmul<<<gridSize, blockSize, 2*k*k*sizeof(float)>>>(dX, dW1, dZ, N, B, k);
    relu<<<1, N*B>>>(dZ, N, B);
    matmul<<<gridSize, blockSize, 2*k*k*sizeof(float)>>>(dZ, dW2, dX, N, B, k);

    // Copy result back to host
    cudaMemcpy(hX, dX, B*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < N; j++) {
            cout << hX[i*N + j] << " ";
        }
        cout << "\n";
    }

    // Cleanup
    cudaFreeHost(hW1); cudaFreeHost(hW2); cudaFreeHost(hX);
    cudaFree(dW1); cudaFree(dW2); cudaFree(dX); cudaFree(dZ);
    
    return 0;
}

