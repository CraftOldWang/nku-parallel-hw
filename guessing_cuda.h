#ifndef GUESSING_CUDA_H
#define GUESSING_CUDA_H

#include "PCFG.h"
#include <cuda_runtime.h>

// CUDA kernels declarations
__global__ void generateGuessesKernel_SingleSegment(
    char** ordered_values,
    int* value_lengths,
    int max_indices,
    char* result_buffer,
    int* result_lengths,
    int max_guess_length
);

__global__ void generateGuessesKernel_MultipleSegments(
    const char* prefix,
    int prefix_length,
    char** ordered_values,
    int* value_lengths,
    int max_indices,
    char* result_buffer,
    int* result_lengths,
    int max_guess_length
);

// Helper functions declarations
char** copyStringVectorToGPU(const vector<string>& strings, int** lengths);
void freeGPUStringMemory(char** d_strings, int n);

// Extended PriorityQueue class with CUDA methods
class PriorityQueue_CUDA : public PriorityQueue {
public:
    // CUDA version of Generate function
    void Generate_CUDA(PT pt);
    
    // CUDA version of PopNext function
    void PopNext_CUDA();
};

#endif // GUESSING_CUDA_H