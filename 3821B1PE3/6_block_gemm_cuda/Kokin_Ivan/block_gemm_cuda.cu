// Copyright (c) 2024 Kokin Ivan

#include <cuda_runtime.h>
#include "block_gemm_cuda.h"

#define TILE_SIZE 16

__global__ void BlockGemmKernel(const float* mxA, const float* mxB,
                                float* mxC, int src) {
    __shared__ float sharedTileA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedTileB[TILE_SIZE][TILE_SIZE];

    int globalRow = blockIdx.y * TILE_SIZE + threadIdx.y;
    int globalCol = blockIdx.x * TILE_SIZE + threadIdx.x;

    float partialSum = 0.0f;

    for (int tileIdx = 0; tileIdx < src / TILE_SIZE; ++tileIdx) {
        sharedTileA[threadIdx.y][threadIdx.x] = mxA[globalRow * src + (tileIdx * TILE_SIZE + threadIdx.x)];
        sharedTileB[threadIdx.y][threadIdx.x] = mxB[(tileIdx * TILE_SIZE + threadIdx.y) * src + globalCol];

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            partialSum += sharedTileA[threadIdx.y][k] * sharedTileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (globalRow < src && globalCol < src) {
        mxC[globalRow * src + globalCol] = partialSum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& hostMxA,
                                 const std::vector<float>& hostMxB,
                                 int src) {
    std::vector<float> hostMxC(src * src, 0.0f);

    float* deviceMxA = nullptr;
    float* deviceMxB = nullptr;
    float* deviceMxC = nullptr;

    size_t mxBytes = src * src * sizeof(float);

    cudaMalloc(&deviceMxA, mxBytes);
    cudaMalloc(&deviceMxB, mxBytes);
    cudaMalloc(&deviceMxC, mxBytes);

    cudaMemcpy(deviceMxA, hostMxA.data(), mxBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMxB, hostMxB.data(), mxBytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((src + TILE_SIZE - 1) / TILE_SIZE, (src + TILE_SIZE - 1) / TILE_SIZE);

    BlockGemmKernel<<<numBlocks, threadsPerBlock>>>(deviceMxA, deviceMxB, deviceMxC, src);

    cudaMemcpy(hostMxC.data(), deviceMxC, mxBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceMxA);
    cudaFree(deviceMxB);
    cudaFree(deviceMxC);

    return hostMxC;
}
