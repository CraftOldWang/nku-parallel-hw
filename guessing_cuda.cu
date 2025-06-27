#include "PCFG.h"
#include "config.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <iostream>

using namespace std;

// CUDA kernel for parallel guess generation (single segment case)
__global__ void generateGuessesKernel_SingleSegment(
    char** ordered_values,
    int* value_lengths,
    int max_indices,
    char* result_buffer,
    int* result_lengths,
    int max_guess_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < max_indices) {
        // Calculate the starting position for this thread's result
        int start_pos = idx * max_guess_length;
        
        // Copy the value to result buffer
        int len = value_lengths[idx];
        for (int i = 0; i < len; i++) {
            result_buffer[start_pos + i] = ordered_values[idx][i];
        }
        result_buffer[start_pos + len] = '\0';
        result_lengths[idx] = len;
    }
}

// CUDA kernel for parallel guess generation (multiple segments case)
__global__ void generateGuessesKernel_MultipleSegments(
    const char* prefix,
    int prefix_length,
    char** ordered_values,
    int* value_lengths,
    int max_indices,
    char* result_buffer,
    int* result_lengths,
    int max_guess_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < max_indices) {
        // Calculate the starting position for this thread's result
        int start_pos = idx * max_guess_length;
        
        // Copy prefix to result buffer
        for (int i = 0; i < prefix_length; i++) {
            result_buffer[start_pos + i] = prefix[i];
        }
        
        // Copy the value to result buffer after prefix
        int value_len = value_lengths[idx];
        for (int i = 0; i < value_len; i++) {
            result_buffer[start_pos + prefix_length + i] = ordered_values[idx][i];
        }
        
        result_buffer[start_pos + prefix_length + value_len] = '\0';
        result_lengths[idx] = prefix_length + value_len;
    }
}

// Helper function to copy string vector to GPU memory
char** copyStringVectorToGPU(const vector<string>& strings, int** lengths) {
    int n = strings.size();
    
    // Allocate memory for lengths array
    *lengths = new int[n];
    for (int i = 0; i < n; i++) {
        (*lengths)[i] = strings[i].length();
    }
    
    // Allocate GPU memory for lengths
    int* d_lengths;
    cudaMalloc(&d_lengths, n * sizeof(int));
    cudaMemcpy(d_lengths, *lengths, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Allocate memory for string pointers
    char** h_strings = new char*[n];
    char** d_strings;
    cudaMalloc(&d_strings, n * sizeof(char*));
    
    // Copy each string to GPU
    for (int i = 0; i < n; i++) {
        char* d_str;
        int len = strings[i].length() + 1;
        cudaMalloc(&d_str, len * sizeof(char));
        cudaMemcpy(d_str, strings[i].c_str(), len * sizeof(char), cudaMemcpyHostToDevice);
        h_strings[i] = d_str;
    }
    
    // Copy string pointers to GPU
    cudaMemcpy(d_strings, h_strings, n * sizeof(char*), cudaMemcpyHostToDevice);
    
    delete[] h_strings;
    return d_strings;
}

// Helper function to free GPU string memory
void freeGPUStringMemory(char** d_strings, int n) {
    char** h_strings = new char*[n];
    cudaMemcpy(h_strings, d_strings, n * sizeof(char*), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n; i++) {
        cudaFree(h_strings[i]);
    }
    cudaFree(d_strings);
    delete[] h_strings;
}

// CUDA version of Generate function for PriorityQueue
void PriorityQueue::Generate_CUDA(PT pt) {
    // Calculate PT probability
    CalProb(pt);
    
    const int MAX_GUESS_LENGTH = 256; // Maximum length for a single guess
    
    if (pt.content.size() == 1) {
        // Single segment case - parallel implementation of first TODO loop
        
        // Find the segment pointer
        segment *a;
        if (pt.content[0].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        else if (pt.content[0].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        else if (pt.content[0].type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        int max_indices = pt.max_indices[0];
        
        // Prepare GPU memory
        int* h_lengths;
        char** d_ordered_values = copyStringVectorToGPU(a->ordered_values, &h_lengths);
        
        int* d_lengths;
        cudaMalloc(&d_lengths, max_indices * sizeof(int));
        cudaMemcpy(d_lengths, h_lengths, max_indices * sizeof(int), cudaMemcpyHostToDevice);
        
        // Allocate result buffer on GPU
        char* d_result_buffer;
        int* d_result_lengths;
        cudaMalloc(&d_result_buffer, max_indices * MAX_GUESS_LENGTH * sizeof(char));
        cudaMalloc(&d_result_lengths, max_indices * sizeof(int));
        
        // Launch CUDA kernel
        int blockSize = 256;
        int gridSize = (max_indices + blockSize - 1) / blockSize;
        
        generateGuessesKernel_SingleSegment<<<gridSize, blockSize>>>(
            d_ordered_values, d_lengths, max_indices,
            d_result_buffer, d_result_lengths, MAX_GUESS_LENGTH
        );
        
        cudaDeviceSynchronize();
        
        // Copy results back to host
        char* h_result_buffer = new char[max_indices * MAX_GUESS_LENGTH];
        int* h_result_lengths = new int[max_indices];
        
        cudaMemcpy(h_result_buffer, d_result_buffer, 
                   max_indices * MAX_GUESS_LENGTH * sizeof(char), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_result_lengths, d_result_lengths, 
                   max_indices * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Convert results to strings and add to guesses
        for (int i = 0; i < max_indices; i++) {
            string guess(h_result_buffer + i * MAX_GUESS_LENGTH, h_result_lengths[i]);
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
        
        // Cleanup
        delete[] h_result_buffer;
        delete[] h_result_lengths;
        delete[] h_lengths;
        cudaFree(d_result_buffer);
        cudaFree(d_result_lengths);
        cudaFree(d_lengths);
        freeGPUStringMemory(d_ordered_values, max_indices);
    }
    else {
        // Multiple segments case - parallel implementation of second TODO loop
        
        // Build prefix string (same as original code)
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1) {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            else if (pt.content[seg_idx].type == 2) {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            else if (pt.content[seg_idx].type == 3) {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1) {
                break;
            }
        }
        
        // Find the last segment pointer
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        else if (pt.content[pt.content.size() - 1].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        else if (pt.content[pt.content.size() - 1].type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        int max_indices = pt.max_indices[pt.content.size() - 1];
        
        // Prepare GPU memory for prefix
        char* d_prefix;
        int prefix_length = guess.length();
        cudaMalloc(&d_prefix, (prefix_length + 1) * sizeof(char));
        cudaMemcpy(d_prefix, guess.c_str(), (prefix_length + 1) * sizeof(char), cudaMemcpyHostToDevice);
        
        // Prepare GPU memory for ordered values
        int* h_lengths;
        char** d_ordered_values = copyStringVectorToGPU(a->ordered_values, &h_lengths);
        
        int* d_lengths;
        cudaMalloc(&d_lengths, max_indices * sizeof(int));
        cudaMemcpy(d_lengths, h_lengths, max_indices * sizeof(int), cudaMemcpyHostToDevice);
        
        // Allocate result buffer on GPU
        char* d_result_buffer;
        int* d_result_lengths;
        cudaMalloc(&d_result_buffer, max_indices * MAX_GUESS_LENGTH * sizeof(char));
        cudaMalloc(&d_result_lengths, max_indices * sizeof(int));
        
        // Launch CUDA kernel
        int blockSize = 256;
        int gridSize = (max_indices + blockSize - 1) / blockSize;
        
        generateGuessesKernel_MultipleSegments<<<gridSize, blockSize>>>(
            d_prefix, prefix_length, d_ordered_values, d_lengths, max_indices,
            d_result_buffer, d_result_lengths, MAX_GUESS_LENGTH
        );
        
        cudaDeviceSynchronize();
        
        // Copy results back to host
        char* h_result_buffer = new char[max_indices * MAX_GUESS_LENGTH];
        int* h_result_lengths = new int[max_indices];
        
        cudaMemcpy(h_result_buffer, d_result_buffer, 
                   max_indices * MAX_GUESS_LENGTH * sizeof(char), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_result_lengths, d_result_lengths, 
                   max_indices * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Convert results to strings and add to guesses
        for (int i = 0; i < max_indices; i++) {
            string temp(h_result_buffer + i * MAX_GUESS_LENGTH, h_result_lengths[i]);
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
        
        // Cleanup
        delete[] h_result_buffer;
        delete[] h_result_lengths;
        delete[] h_lengths;
        cudaFree(d_prefix);
        cudaFree(d_result_buffer);
        cudaFree(d_result_lengths);
        cudaFree(d_lengths);
        freeGPUStringMemory(d_ordered_values, max_indices);
    }
}

// CUDA version of PopNext function
void PriorityQueue::PopNext_CUDA() {
    // Generate guesses using CUDA for the front PT
    Generate_CUDA(priority.front());
    
    // Generate new PTs (same as original)
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts) {
        // Calculate probability
        CalProb(pt);
        // Insert new PT into priority queue based on probability
        for (auto iter = priority.begin(); iter != priority.end(); iter++) {
            if (iter != priority.end() - 1 && iter != priority.begin()) {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob) {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1) {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob) {
                priority.emplace(iter, pt);
                break;
            }
        }
    }
    
    // Remove the front PT from queue
    priority.erase(priority.begin());
}