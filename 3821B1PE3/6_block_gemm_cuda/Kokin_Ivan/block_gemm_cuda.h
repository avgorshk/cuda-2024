// Copyright (c) 2024 Kokin Ivan

#ifndef __BLOCK_GEMM_CUDA_H
#define __BLOCK_GEMM_CUDA_H

#include <vector>

std::vector<float> BlockGemmCUDA(const float* mxA,
                                 const float* mxB,
                                 float* mxC,int src);

#endif // __BLOCK_GEMM_CUDA_H