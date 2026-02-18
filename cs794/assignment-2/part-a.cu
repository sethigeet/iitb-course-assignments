#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void matmul(float* A, float* B, float* C, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < N && y < N) {
        float res = 0;
        for (int i = 0; i < N; i++) {
            res += A[y*N + i] * B[i*N + x];
        }
        C[y*N + x] = res;
    }
}

#define N 40
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
        }
    }
    
    // Copy matrices to device
    cudaMemcpy(dA, hA, MSIZE(N), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, MSIZE(N), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, N);
    
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

