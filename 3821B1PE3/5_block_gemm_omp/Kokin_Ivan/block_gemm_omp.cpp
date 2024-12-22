// Copyright (c) 2024 Kokin Ivan

#include "block_gemm_omp.h"

#define BLOCK_SIZE 16

std::vector<float> BlockGemmOMP(const std::vector<float>& mxA,
    const std::vector<float>& mxB,
    int src) {
    std::vector<float> mxC(src * src, 0.0f);
    int totalBlocks = src / BLOCK_SIZE;

#pragma omp parallel for collapse(2)
    for (int blcR = 0; blcR < totalBlocks; ++blcR) {
        for (int blcC = 0; blcC < totalBlocks; ++blcC) {
            for (int blcInd = 0; blcInd < totalBlocks; ++blcInd) {
                for (int lclR = 0; lclR < BLOCK_SIZE; ++lclR) {
                    for (int lclC = 0; lclC < BLOCK_SIZE; ++lclC) {
                        float lclSum = 0.0f;
                        for (int localIndex = 0; localIndex < BLOCK_SIZE; ++localIndex) {
                            lclSum += mxA[(blcR * BLOCK_SIZE + lclR) * src + blcInd * BLOCK_SIZE + localIndex] *
                                mxB[(blcInd * BLOCK_SIZE + localIndex) * src + blcC * BLOCK_SIZE + lclC];
                        }
                        mxC[(blcR * BLOCK_SIZE + lclR) * src + blcC * BLOCK_SIZE + lclC] += lclSum;
                    }
                }
            }
        }
    }
    return mxC;
}
