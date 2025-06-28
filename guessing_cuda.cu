#include "guessing_cuda.h"
#include "PCFG.h"
#include "config.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <iostream>
#include <numeric>

using namespace std;

// Device function for upper_bound (binary search). Finds the first element > val.
__device__ int upper_bound_device(const int* arr, int n, int val) {
    int low = 0;
    int high = n;
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (val >= arr[mid]) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

// Unified CUDA kernel for batch processing
__global__ void generateGuessesKernel_Batch(
    // Task metadata
    int num_tasks,
    const int* task_offsets, // Cumulative guess counts, e.g., [0, 100, 150, ...]

    // Flattened data for all tasks
    const char* all_prefixes,
    const int* prefix_lengths,
    const int* prefix_offsets,
    const char* all_values,
    const int* value_lengths,
    const int* value_offsets_flat,
    const int* task_value_offsets, // Maps task_idx to the start of its values in the value arrays

    // Output buffers
    char* result_buffer,
    int* result_lengths,
    int max_guess_length
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= task_offsets[num_tasks]) {
        return;
    }

    // 1. Find the task this thread belongs to using binary search
    int task_idx = upper_bound_device(task_offsets, num_tasks + 1, global_idx) - 1;

    // 2. Calculate the index within the task's value segment
    int local_idx = global_idx - task_offsets[task_idx];

    // 3. Get prefix info
    int prefix_len = prefix_lengths[task_idx];
    int prefix_start = prefix_offsets[task_idx];

    // 4. Get value info
    int value_block_start_idx = task_value_offsets[task_idx];
    int global_value_idx = value_block_start_idx + local_idx;
    int value_len = value_lengths[global_value_idx];
    int value_start = value_offsets_flat[global_value_idx];

    // 5. Assemble the guess string in the result buffer
    int result_start_pos = global_idx * max_guess_length;
    int current_pos = result_start_pos;

    // Copy prefix
    for (int i = 0; i < prefix_len; i++) {
        result_buffer[current_pos++] = all_prefixes[prefix_start + i];
    }

    // Copy value
    for (int i = 0; i < value_len; i++) {
        result_buffer[current_pos++] = all_values[value_start + i];
    }

    // Set final length
    result_lengths[global_idx] = prefix_len + value_len;
}


// --- CUDABatchManager Implementation ---

CUDABatchManager::CUDABatchManager(vector<string>& guesses_ref, long long& total_guesses_ref)
    : guesses(guesses_ref), total_guesses(total_guesses_ref) {}

CUDABatchManager::~CUDABatchManager() {
    if (!tasks.empty()) {
        flush();
    }
}

void CUDABatchManager::addTask(const GuessTask& task) {
    tasks.push_back(task);
    if (tasks.size() >= CUDA_BATCH_THRESHOLD) {
        flush();
    }
}

void CUDABatchManager::flush() {
    if (tasks.empty()) {
        return;
    }

    // --- 1. Flatten Data on CPU ---
    vector<int> h_prefix_lengths, h_prefix_offsets;
    vector<char> h_all_prefixes;

    vector<int> h_value_lengths, h_value_offsets_flat, h_task_value_offsets;
    vector<char> h_all_values;
    
    vector<int> h_task_offsets; // Cumulative guess counts for kernel indexing
    int total_guess_count = 0;

    int current_prefix_offset = 0;
    int current_value_block_offset = 0; // The start index for the current task's values
    int current_flat_value_offset = 0;  // The start position for the current value string in h_all_values

    h_task_offsets.push_back(0);

    for (const auto& task : tasks) {
        // Prefix
        h_prefix_lengths.push_back(task.prefix.length());
        h_prefix_offsets.push_back(current_prefix_offset);
        h_all_prefixes.insert(h_all_prefixes.end(), task.prefix.begin(), task.prefix.end());
        current_prefix_offset += task.prefix.length();

        // Values
        h_task_value_offsets.push_back(current_value_block_offset);
        for(int i = 0; i < task.max_indices; ++i) {
            const string& val = task.ordered_values_ptr->at(i);
            h_value_lengths.push_back(val.length());
            h_value_offsets_flat.push_back(current_flat_value_offset);
            h_all_values.insert(h_all_values.end(), val.begin(), val.end());
            current_flat_value_offset += val.length();
        }
        current_value_block_offset += task.max_indices;

        // Task offsets (cumulative guess count)
        total_guess_count += task.max_indices;
        h_task_offsets.push_back(total_guess_count);
    }

    // --- 2. Allocate and Copy Memory to GPU ---
    int* d_task_offsets;
    cudaMalloc(&d_task_offsets, h_task_offsets.size() * sizeof(int));
    cudaMemcpy(d_task_offsets, h_task_offsets.data(), h_task_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    char* d_all_prefixes;
    cudaMalloc(&d_all_prefixes, h_all_prefixes.size() * sizeof(char));
    cudaMemcpy(d_all_prefixes, h_all_prefixes.data(), h_all_prefixes.size() * sizeof(char), cudaMemcpyHostToDevice);

    int* d_prefix_lengths;
    cudaMalloc(&d_prefix_lengths, h_prefix_lengths.size() * sizeof(int));
    cudaMemcpy(d_prefix_lengths, h_prefix_lengths.data(), h_prefix_lengths.size() * sizeof(int), cudaMemcpyHostToDevice);

    int* d_prefix_offsets;
    cudaMalloc(&d_prefix_offsets, h_prefix_offsets.size() * sizeof(int));
    cudaMemcpy(d_prefix_offsets, h_prefix_offsets.data(), h_prefix_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    char* d_all_values;
    cudaMalloc(&d_all_values, h_all_values.size() * sizeof(char));
    cudaMemcpy(d_all_values, h_all_values.data(), h_all_values.size() * sizeof(char), cudaMemcpyHostToDevice);

    int* d_value_lengths;
    cudaMalloc(&d_value_lengths, h_value_lengths.size() * sizeof(int));
    cudaMemcpy(d_value_lengths, h_value_lengths.data(), h_value_lengths.size() * sizeof(int), cudaMemcpyHostToDevice);

    int* d_value_offsets_flat;
    cudaMalloc(&d_value_offsets_flat, h_value_offsets_flat.size() * sizeof(int));
    cudaMemcpy(d_value_offsets_flat, h_value_offsets_flat.data(), h_value_offsets_flat.size() * sizeof(int), cudaMemcpyHostToDevice);

    int* d_task_value_offsets;
    cudaMalloc(&d_task_value_offsets, h_task_value_offsets.size() * sizeof(int));
    cudaMemcpy(d_task_value_offsets, h_task_value_offsets.data(), h_task_value_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Output buffers
    char* d_result_buffer;
    int* d_result_lengths;
    cudaMalloc(&d_result_buffer, total_guess_count * CUDA_MAX_GUESS_LENGTH * sizeof(char));
    cudaMalloc(&d_result_lengths, total_guess_count * sizeof(int));

    // --- 3. Launch Kernel ---
    int blockSize = CUDA_BLOCK_SIZE;
    int gridSize = (total_guess_count + blockSize - 1) / blockSize;
    
    generateGuessesKernel_Batch<<<gridSize, blockSize>>>(
        tasks.size(), d_task_offsets,
        d_all_prefixes, d_prefix_lengths, d_prefix_offsets,
        d_all_values, d_value_lengths, d_value_offsets_flat, d_task_value_offsets,
        d_result_buffer, d_result_lengths, CUDA_MAX_GUESS_LENGTH
    );
    cudaDeviceSynchronize();

    // --- 4. Copy Results Back to Host ---
    vector<char> h_result_buffer(total_guess_count * CUDA_MAX_GUESS_LENGTH);
    vector<int> h_result_lengths(total_guess_count);
    cudaMemcpy(h_result_buffer.data(), d_result_buffer, h_result_buffer.size() * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result_lengths.data(), d_result_lengths, h_result_lengths.size() * sizeof(int), cudaMemcpyDeviceToHost);

    // --- 5. Process Results ---
    guesses.reserve(guesses.size() + total_guess_count);
    for (int i = 0; i < total_guess_count; ++i) {
        guesses.emplace_back(h_result_buffer.data() + i * CUDA_MAX_GUESS_LENGTH, h_result_lengths[i]);
    }
    total_guesses += total_guess_count;

    // --- 6. Cleanup GPU Memory ---
    cudaFree(d_task_offsets);
    cudaFree(d_all_prefixes);
    cudaFree(d_prefix_lengths);
    cudaFree(d_prefix_offsets);
    cudaFree(d_all_values);
    cudaFree(d_value_lengths);
    cudaFree(d_value_offsets_flat);
    cudaFree(d_task_value_offsets);
    cudaFree(d_result_buffer);
    cudaFree(d_result_lengths);

    // --- 7. Clear CPU Task Buffer ---
    tasks.clear();
}


// --- PriorityQueue_CUDA Implementation ---

PriorityQueue_CUDA::PriorityQueue_CUDA(PCFG &pcfg)
    : m(pcfg), batchManager(guesses, total_guesses), total_guesses(0) {
    priority = pcfg.get_initial_PT();
}

PriorityQueue_CUDA::~PriorityQueue_CUDA() {
    // The batchManager's destructor will handle any remaining tasks.
}

void PriorityQueue_CUDA::PopNext_CUDA() {
    if (priority.empty()) {
        return;
    }
    
    PT pt = priority.front();
    CalProb(pt); // Ensure probability is calculated

    GuessTask task;
    segment* last_segment;

    if (pt.content.size() == 1) {
        task.prefix = "";
    } else {
        string p_str;
        for (size_t i = 0; i < pt.content.size() - 1; ++i) {
            p_str += m.get_segment_value(pt.content[i], pt.curr_indices[i]);
        }
        task.prefix = p_str;
    }

    const auto& last_seg_info = pt.content.back();
    if (last_seg_info.type == 1) last_segment = &m.letters[m.FindLetter(last_seg_info)];
    else if (last_seg_info.type == 2) last_segment = &m.digits[m.FindDigit(last_seg_info)];
    else last_segment = &m.symbols[m.FindSymbol(last_seg_info)];

    task.ordered_values_ptr = &last_segment->ordered_values;
    task.max_indices = pt.max_indices.back();

    batchManager.addTask(task);

    // Generate new PTs and update priority queue (CPU-side logic)
    vector<PT> new_pts = priority.front().NewPTs();
    priority.erase(priority.begin());
    for (PT new_pt : new_pts) {
        CalProb(new_pt);
        auto it = std::lower_bound(priority.begin(), priority.end(), new_pt, 
            [](const PT& a, const PT& b) {
            return a.prob > b.prob;
        });
        priority.insert(it, new_pt);
    }
}

void PriorityQueue_CUDA::Flush_CUDA() {
    batchManager.flush();
}

