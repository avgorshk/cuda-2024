#include <iostream>
#include <vector>
#include <chrono>
#include "gelu_cuda.h"

int main() {
    const size_t size = 1000000; // Размер входного вектора
    std::vector<float> input(size, 1.0f); // Пример входных данных

    // Warming-up
    GeluCUDA(input);

    // Performance Measuring
    auto start = std::chrono::high_resolution_clock::now();
    auto result = GeluCUDA(input);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    return 0;
}
