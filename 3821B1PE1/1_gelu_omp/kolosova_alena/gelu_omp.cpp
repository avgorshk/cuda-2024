#include "gelu_omp.h"
#include <omp.h>
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const int size = input.size();

    std::vector<float> output(size);

    const float geluConst1 = 0.044715f;
    const float geluConst2 = 0.7978845608f;

#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + tanh(geluConst1 * (x + geluConst2 * x * x * x)));
    }

    return output;
}