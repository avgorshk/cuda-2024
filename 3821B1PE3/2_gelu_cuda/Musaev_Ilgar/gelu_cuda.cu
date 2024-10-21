// Copyright (c) 2024 Musaev Ilgar
#include "gelu_cuda.h"
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

// Define a macro for the constant factor
#define GELU_CONSTANT (0.5f * (2.0f / sqrt(3.14f)))

// Define a macro for the cubic term coefficient
#define CUBIC_TERM_COEFFICIENT (0.044715f)

// Define a kernel for the GELU activation function
__global__ void geluKernel(const float* input, float* output, const int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = input[i];
    // Calculate the intermediate result efficiently
    float intermediate = GELU_CONSTANT * (x + CUBIC_TERM_COEFFICIENT * x * x * x);
    // Apply tanh and multiply by x
    output[i] = 0.5f * x * (1.0f + tanh(intermediate));
  }
}

// Implement the GeluCUDA function
std::vector<float> GeluCUDA(const std::vector<float>& input) {
  int n = input.size();
  std::vector<float> output(n);

  // Allocate device memory
  float* d_input;
  float* d_output;
  cudaMalloc(&d_input, n * sizeof(float));
  cudaMalloc(&d_output, n * sizeof(float));

  // Copy input data to device
  cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel with a larger block size
  int threadsPerBlock = 512; // Experiment with different block sizes for optimal performance
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  geluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);

  // Copy output data from device
  cudaMemcpy(output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_output);

  return output;
}
