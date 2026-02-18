#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void matmul(const float* A, const float* B, float* C, const int N) {
    const int tileSize = blockDim.x;
    const int numTiles = gridDim.x;
    
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int localX = threadIdx.x;
    int localY = threadIdx.y;
    
    extern __shared__ float tiles[];
    float* tileA = tiles;
    float* tileB = tiles + (tileSize*tileSize);


    for (int tileIdx = 0; tileIdx < numTiles; tileIdx++) {
        int Ax = tileIdx * tileSize + localX;
        int Ay = globalY;
        
        int Bx = globalX;
        int By = tileIdx * tileSize + localY;
        
        if (Ax < N && Ay < N) {
            tileA[localY*tileSize + localX] = A[Ay*N + Ax];
        } else {
            tileA[localY*tileSize + localX] = 0;
        }
        
        if (Bx < N && By < N) {
            tileB[localY*tileSize + localX] = B[By*N + Bx];
        } else {
            tileB[localY*tileSize + localX] = 0;
        }

        __syncthreads();

        float res = 0;
        for (int i = 0; i < tileSize; i++) {
            res += tileA[localY*tileSize + i] * tileB[i*tileSize + localX];
        }

        C[globalY*N + globalX] += res;

        __syncthreads();
    }
}

#define N 4
#define tileSize 2
#define MSIZE(n) ((n) * (n) * sizeof(float))

int main() {
    float *hA, *dA;
    float *hB, *dB;
    float *hC, *dC;
    
    // Allocate memory for matrices
    hA = (float*)malloc(MSIZE(N));
    cudaMalloc(&dA, MSIZE(N));
    hB = (float*)malloc(MSIZE(N));
    cudaMalloc(&dB, MSIZE(N));
    hC = (float*)malloc(MSIZE(N));
    cudaMalloc(&dC, MSIZE(N));

    // Fill matrices with elements
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                hA[i*N + j] = 1;
            } else {
                hA[i*N + j] = 0;
            }
            
            hB[i*N + j] = i+j;
            hC[i*N + j] = 0;
        }
    }
    
    // Copy matrices to device
    cudaMemcpy(dA, hA, MSIZE(N), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, MSIZE(N), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, MSIZE(N), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(tileSize, tileSize);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    constexpr unsigned int sharedMemSize = 2 * MSIZE(tileSize);
    matmul<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(dA, dB, dC, N);
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result to host
    cudaMemcpy(hC, dC, MSIZE(N), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << hC[i*N + j] << " ";
        }
        cout << "\n";
    }

    // Cleanup
    free(hA); free(hB); free(hC);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    
    return 0;
}

