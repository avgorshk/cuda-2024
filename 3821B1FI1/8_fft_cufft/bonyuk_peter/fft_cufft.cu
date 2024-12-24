/*When entering the following vector:
int batch = 1;
    std::vector<float> input = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0}; 
The output values ​​were:
1 0 2 0 3 0 4 0 
*/

#include "fft_cufft.h"
#include <cufft.h>
#include <iostream>

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);

    cufftComplex* d_input;
    cufftComplex* d_output;
    std::vector<float> output(2 * n * batch);

    cudaMalloc((void**)&d_input, sizeof(cufftComplex) * n * batch);
    cudaMalloc((void**)&d_output, sizeof(cufftComplex) * n * batch);

    cudaMemcpy(d_input, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD);

    cufftExecC2C(plan, d_output, d_input, CUFFT_INVERSE);

    float normalizationFactor = 1.0f / n;
    cudaMemcpy(output.data(), d_input, sizeof(float) * output.size(), cudaMemcpyDeviceToHost);
    for (int i = 0; i < output.size(); ++i) {
        output[i] *= normalizationFactor;
    }

    cufftDestroy(plan);
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
