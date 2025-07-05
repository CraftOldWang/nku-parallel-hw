#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <string_view>
#include <mutex>
#include "guessing_cuda.h"  // 包含头文件
#include "config.h"
#include <chrono>
using namespace std;
using namespace chrono;

// CUDA错误检查宏
// gemini 改进过的
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            fprintf(stderr, "\033[1;31m[CUDA ERROR]\033[0m %s:%d: %s (%d)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err__), err__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


GpuOrderedValuesData* gpu_data = nullptr;
TaskManager* task_manager = nullptr;
SegmentLengthMaps* SegmentLengthMaps::instance = nullptr;
PTMaps* PTMaps::instance = nullptr;

// 统一的缓冲区管理（无论是否使用线程池）

void async_gpu_task(AsyncGpuTask* task_data, PriorityQueue& q) {
#ifdef DEBUG
    printf("[DEBUG] 🎯 async_gpu_task: Received async GPU task with %d tasks, %d guesses\n", 
           task_data->task_manager.taskcount, task_data->task_manager.guesscount);
#endif

    
    try {
#ifdef DEBUG
        printf("[DEBUG] 🎯 async_gpu_task: Launching async pipeline...\n");
#endif


        AsyncGpuPipeline::launch_async_pipeline(std::move(task_data->task_manager), q);
#ifdef DEBUG
    printf("[DEBUG] 🎯 async_gpu_task: Pipeline launched successfully\n");

#endif        
    } catch (const std::exception& e) {
#ifdef DEBUG
        printf("[DEBUG] ❌ async_gpu_task: Pipeline launch failed: %s\n", e.what());
#endif
        std::cerr << "GPU async pipeline error: " << e.what() << std::endl;
        
    }
    
    // 并不清理，， 因为 task_manager d的生命周期是 后续流水线管理了（被移交控制权了）
    // 清理AsyncGpuTask
    // task_data里面的task_manager 已经被拿走了， 故释放
    delete task_data;


#ifdef DEBUG
    printf("[DEBUG] 🎯 async_gpu_task: Task data cleaned up\n");
#endif


}


#ifdef TIME_COUNT
double time_add_task = 0;
double time_launch_task = 0;
double time_before_launch = 0;
double time_after_launch = 0;
double time_all_batch = 0;
double time_string_process = 0;
double time_memcpy_toh = 0;
#endif




// GPU设备函数：使用二分查找找到guess_id对应的task
__device__ int find_task_for_guess(Taskcontent* d_tasks, int guess_id, int& task_start, int& task_end) {
    int left = 0, right = d_tasks->taskcount - 1;
    int found_task_id = -1;
    
    while (left <= right) {
        int mid = (left + right) / 2;
        
        // 使用预计算的累积偏移数组
        int mid_start = d_tasks->cumulative_guess_offsets[mid];
        int mid_end = d_tasks->cumulative_guess_offsets[mid + 1];
        
        if (guess_id >= mid_start && guess_id < mid_end) {
            // 找到了对应的task
            found_task_id = mid;
            task_start = mid_start;
            task_end = mid_end;
            break;
        } else if (guess_id < mid_start) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    return found_task_id;
}



// guess的数量 > GPU_BATCH_SIZE 个 。 kernal函数
//TODO 核函数中间逻辑有点。。。。也许还能优化。
//TODO 傻了， 我这个一次性最多 10_0000 ~ 100_0000 个猜测， 每个block最多1024 个线程， 但是
// 数据如下， blockDimx 可以特别大.... 麻了
//每个block最大线程数: 1024
//每个SM最大线程数: 1024
// block维度限制: (1024, 1024, 64)
// grid维度限制: (2147483647, 65535, 65535)
// 共有SM数量: 40
__global__ void generate_guesses_kernel(
    GpuOrderedValuesData* gpu_data,
    Taskcontent* d_tasks,
    char* d_guess_buffer
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // 计算每个线程处理的guess块大小
    int guesses_per_thread = (d_tasks->guesscount + total_threads - 1) / total_threads;
    
    // 当前线程处理的guess范围
    int start_guess = tid * guesses_per_thread;
    int end_guess = min(start_guess + guesses_per_thread, d_tasks->guesscount);
    
    // 缓存当前task的信息 
    int current_task_id = -1;
    int current_task_start = 0;
    int current_task_end = 0;
    
    // 缓存当前task的segment信息
    int seg_type, seg_id, seg_len, prefix_len, prefix_offset;
    int task_output_offset;
    char* all_values;
    int* value_offsets;
    int* seg_offsets;
    int seg_start_idx;
    
    for (int guess_id = start_guess; guess_id < end_guess; guess_id++) {
        
        // 检查是否需要更新task信息
        if (guess_id < current_task_start || guess_id >= current_task_end) {
            // 使用设备函数进行二分查找
            current_task_id = find_task_for_guess(d_tasks, guess_id, current_task_start, current_task_end);
            
            if (current_task_id != -1) {
                // 缓存task信息
                seg_type = d_tasks->seg_types[current_task_id];
                seg_id = d_tasks->seg_ids[current_task_id];
                seg_len = d_tasks->seg_lens[current_task_id];
                prefix_len = d_tasks->prefix_lens[current_task_id];
                prefix_offset = d_tasks->prefix_offsets[current_task_id];
                task_output_offset = d_tasks->output_offsets[current_task_id];
                
                // 选择数据源
                if (seg_type == 1) {
                    all_values = gpu_data->letter_all_values;
                    value_offsets = gpu_data->letter_value_offsets;
                    seg_offsets = gpu_data->letter_seg_offsets;
                } else if (seg_type == 2) {
                    all_values = gpu_data->digit_all_values;
                    value_offsets = gpu_data->digits_value_offsets;
                    seg_offsets = gpu_data->digit_seg_offsets;
                } else {
                    all_values = gpu_data->symbol_all_values;
                    value_offsets = gpu_data->symbol_value_offsets;
                    seg_offsets = gpu_data->symbol_seg_offsets;
                }
                
                seg_start_idx = seg_offsets[seg_id];
            }
        }

        // 当前guess在task中的局部索引
        int local_guess_idx = guess_id - current_task_start;
        
        // 找到对应的value
        int value_idx = seg_start_idx + local_guess_idx;
        
        // 计算输出位置
        int output_offset = task_output_offset + local_guess_idx * (seg_len + prefix_len);
        
        // 复制前缀
        for (int i = 0; i < prefix_len; i++) {
            d_guess_buffer[output_offset + i] = d_tasks->prefixs[prefix_offset + i];
        }
        
        // 复制value
        int value_start = value_offsets[value_idx];
        
        for (int i = 0; i < seg_len; i++) {
            d_guess_buffer[output_offset + prefix_len + i] = all_values[value_start + i];
        }
    }
}


void TaskManager::add_task(const segment* seg, string& prefix, PriorityQueue& q) {

#ifdef TIME_COUNT
auto start_add_task = system_clock::now();
#endif
    // 获得映射表
    SegmentLengthMaps* maps = SegmentLengthMaps::getInstance();
#ifdef DEBUG
    if (seg == nullptr) {
        cout << "what the fuck ?" << endl;
        std::this_thread::sleep_for(std::chrono::seconds(10000)); // 睡 1000 秒

    }
#endif
    seg_types.push_back(seg->type);
    seg_lens.push_back(seg->length);
    const segment & seginmodel = maps->getSeginPQ(*seg, q); 
    seg_ids.push_back(maps->getID(*seg));

    prefix_lens.push_back(prefix.length());
    // prefixs.push_back(prefix);  // 先别使用移动语义
    prefixs.emplace_back(std::move(prefix));  // 使用移动语义

    taskcount++;
    seg_value_count.push_back(seginmodel.ordered_values.size());  
    guesscount += seginmodel.ordered_values.size();

#ifdef DEBUG
    // cout << "Added task: ";
    // seginmodel.PrintSeg();
    // cout << " -> ID: " << seg_ids.back() << endl;
#endif


#ifdef TIME_COUNT
auto end_add_task = system_clock::now();
auto duration_add_task = duration_cast<microseconds>(end_add_task - start_add_task);
time_add_task += double(duration_add_task.count()) * microseconds::period::num / microseconds::period::den;
#endif

}

void SegmentLengthMaps::init(PriorityQueue& q) {
    if (initialized) return;
    
    // 构建字母长度映射
    for (int i = 0; i < q.m.letters.size(); i++) {
        int length = q.m.letters[i].length;
        if (letter_length_to_id.find(length) == letter_length_to_id.end()) {
            letter_length_to_id[length] = i;
        }
    }
    
    // 构建数字长度映射
    for (int i = 0; i < q.m.digits.size(); i++) {
        int length = q.m.digits[i].length;
        if (digit_length_to_id.find(length) == digit_length_to_id.end()) {
            digit_length_to_id[length] = i;
        }
    }
    
    // 构建符号长度映射
    for (int i = 0; i < q.m.symbols.size(); i++) {
        int length = q.m.symbols[i].length;
        if (symbol_length_to_id.find(length) == symbol_length_to_id.end()) {
            symbol_length_to_id[length] = i;
        }
    }
    
    initialized = true;
#ifdef DEBUG
cout << "Length mapping initialization completed:" << endl;
cout << "  Letter length mappings: " << letter_length_to_id.size() << " types" << endl;
cout << "  Digit length mappings: " << digit_length_to_id.size() << " types" << endl;
cout << "  Symbol length mappings: " << symbol_length_to_id.size() << " types" << endl;
#endif
}


void TaskManager::clean() {
    // Clean data in TaskManager
    seg_types.clear();
    seg_ids.clear();
    seg_lens.clear();
    prefixs.clear();
    prefix_lens.clear();
    taskcount = 0;
    guesscount = 0;
    seg_value_count.clear();

}


void TaskManager::print() {
    cout << "TaskManager state:" << endl;
    cout << "Total tasks: " << taskcount << endl;
    cout << "Total guesses: " << guesscount << endl;
    cout << "Segment types: ";
    for (const auto& type : seg_types) {
        cout << type << " ";
    }
    cout << endl;

    cout << "Segment IDs: ";
    for (const auto& id : seg_ids) {
        cout << id << " ";
    }
    cout << endl;

    cout << "Segment lengths: ";
    for (const auto& len : seg_lens) {
        cout << len << " ";
    }
    cout << endl;

    cout << "Prefixes: ";
    for (const auto& prefix : prefixs) {
        cout << prefix << " ";
    }
    cout << endl;

    cout << "Prefix lengths: ";
    for (const auto& len : prefix_lens) {
        cout << len << " ";
    }
    cout << endl;

}



void init_gpu_ordered_values_data(GpuOrderedValuesData*& d_gpu_data,PriorityQueue& q) {

    //cpu上的数据
    GpuOrderedValuesData h_gpu_data;

    // Calculate total length of char* arrays
    size_t total_letter_length = 0;
    size_t total_digit_length = 0;
    size_t total_symbol_length = 0;

    // Number of values of each type 
    size_t letter_offsetarr_length = 0;
    size_t digit_offsetarr_length = 0;
    size_t symbol_offsetarr_length = 0;

    char* letter_all_values = nullptr;// Flatten ordered_values of each segment
    char* digit_all_values= nullptr;
    char* symbol_all_values= nullptr;

    int* letter_value_offsets= nullptr; // Start position of each ordered_value in letter_all_values
    int* digits_value_offsets= nullptr; // Start position of each ordered_value in digit_all_values
    int* symbol_value_offsets= nullptr; // Start position of each ordered_value in symbol_all_values

    int* letter_seg_offsets= nullptr; // Which ordered_value is the first one of each letter segment in value_offsets
    int* digit_seg_offsets= nullptr; // Which ordered_value is the first one of each digit segment in value_offsets
    int* symbol_seg_offsets= nullptr; // Which ordered_value is the first one of each symbol segment in value_offsets
    for (const auto& seg : q.m.letters) {
        total_letter_length += seg.ordered_values.size() * seg.length;
        letter_offsetarr_length += seg.ordered_values.size();
    }
    for (const auto& seg : q.m.digits) {
        total_digit_length += seg.ordered_values.size()* seg.length;
        digit_offsetarr_length += seg.ordered_values.size();
    }
    for (const auto& seg : q.m.symbols) {
        total_symbol_length += seg.ordered_values.size()* seg.length;
        symbol_offsetarr_length += seg.ordered_values.size();
    }

    letter_all_values = new char[total_letter_length];
    letter_value_offsets = new int[letter_offsetarr_length+1];
    letter_seg_offsets = new int[q.m.letters.size()+1];

    digit_all_values = new char[total_digit_length];
    digits_value_offsets = new int[digit_offsetarr_length+1];
    digit_seg_offsets = new int[q.m.digits.size()+1];

    symbol_all_values = new char[total_symbol_length];
    symbol_value_offsets = new int[symbol_offsetarr_length+1];
    symbol_seg_offsets = new int[q.m.symbols.size()+1]; 

    // Added one more offset to represent the end, actually not corresponding to segment and value

    int value_offset = 0;
    int seg_offset  = 0;

    for(int i = 0 ;i < q.m.letters.size() ;i++) {
        letter_seg_offsets[i] = seg_offset;
        const auto& seg = q.m.letters[i];
        for (int j =0 ;j < seg.ordered_values.size();j++ ){
            letter_value_offsets[seg_offset++] = value_offset;
            string value = seg.ordered_values[j];
            for(int k =0; k < value.length(); k++) {
                letter_all_values[value_offset++] = value[k];
            }
        }
    }
    letter_seg_offsets[q.m.letters.size()] = seg_offset; // Fill in the last one... actually corresponds to length 0
    letter_value_offsets[letter_offsetarr_length] = value_offset;
    seg_offset = 0;
    value_offset = 0;

    for (int i=0;i< q.m.digits.size();i++) {
        digit_seg_offsets[i] = seg_offset;
        const auto& seg = q.m.digits[i];
        for (int j=0; j< seg.ordered_values.size();j++ ) {
            digits_value_offsets[seg_offset++] = value_offset;
            string value = seg.ordered_values[j];
            for(int k = 0;k < value.length(); k++) {
                digit_all_values[value_offset++] = value[k];
            }
        }
    }
    digit_seg_offsets[q.m.digits.size()] = seg_offset; // Fill in the last one... actually corresponds to length 0
    digits_value_offsets[digit_offsetarr_length] = value_offset;
    seg_offset = 0;
    value_offset = 0;   
    for (int i=0;i< q.m.symbols.size();i++) {
        symbol_seg_offsets[i] = seg_offset;
        const auto& seg = q.m.symbols[i];
        for (int j=0; j< seg.ordered_values.size();j++ ) {
            symbol_value_offsets[seg_offset++] = value_offset;
            string value = seg.ordered_values[j];
            for(int k = 0;k < value.length(); k++) {
                symbol_all_values[value_offset++] = value[k];
            }
        }
    }
    symbol_seg_offsets[q.m.symbols.size()] = seg_offset; // Fill in the last one
    symbol_value_offsets[symbol_offsetarr_length] = value_offset;

    // Related things need addresses... pointers are just pointers, interpreted as CPU or GPU memory depending on specific context
    // For example, cudaMemcpy uses cudaMemcpyKind kind to distinguish.

    // Copy data of each pointer to GPU


    // Allocate memory and copy
    CUDA_CHECK(cudaMalloc(&h_gpu_data.letter_all_values, total_letter_length * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&h_gpu_data.digit_all_values, total_digit_length * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&h_gpu_data.symbol_all_values, total_symbol_length * sizeof(char)));
    CUDA_CHECK(cudaMemcpy(h_gpu_data.letter_all_values, letter_all_values, total_letter_length * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_gpu_data.digit_all_values, digit_all_values, total_digit_length * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_gpu_data.symbol_all_values, symbol_all_values, total_symbol_length * sizeof(char), cudaMemcpyHostToDevice));

    // Allocate offset arrays and copy
    CUDA_CHECK(cudaMalloc(&h_gpu_data.letter_value_offsets, (letter_offsetarr_length+1)* sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h_gpu_data.digits_value_offsets, (digit_offsetarr_length+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h_gpu_data.symbol_value_offsets, (symbol_offsetarr_length+1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(h_gpu_data.letter_value_offsets, letter_value_offsets, (letter_offsetarr_length+1)* sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_gpu_data.digits_value_offsets, digits_value_offsets, (digit_offsetarr_length+1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_gpu_data.symbol_value_offsets, symbol_value_offsets, (symbol_offsetarr_length+1) * sizeof(int), cudaMemcpyHostToDevice));


    // Allocate segment offset arrays and copy
    CUDA_CHECK(cudaMalloc(&h_gpu_data.letter_seg_offsets, q.m.letters.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h_gpu_data.digit_seg_offsets, q.m.digits.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h_gpu_data.symbol_seg_offsets, q.m.symbols.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(h_gpu_data.letter_seg_offsets, letter_seg_offsets, q.m.letters.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_gpu_data.digit_seg_offsets, digit_seg_offsets, q.m.digits.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_gpu_data.symbol_seg_offsets, symbol_seg_offsets, q.m.symbols.size() * sizeof(int), cudaMemcpyHostToDevice));


    //Copy struct to GPU
    CUDA_CHECK(cudaMalloc(&d_gpu_data, sizeof(GpuOrderedValuesData)));
    CUDA_CHECK(cudaMemcpy(d_gpu_data, &h_gpu_data, sizeof(GpuOrderedValuesData), cudaMemcpyHostToDevice));

    // Release CPU-side temporary memory
    delete[] letter_all_values;
    delete[] digit_all_values;
    delete[] symbol_all_values;
    delete[] letter_value_offsets;
    delete[] digits_value_offsets;
    delete[] symbol_value_offsets;
    delete[] letter_seg_offsets;
    delete[] digit_seg_offsets;
    delete[] symbol_seg_offsets;


}

void clean_gpu_ordered_values_data(GpuOrderedValuesData *& d_gpu_data){
    if (d_gpu_data == nullptr) return;
    

    // Copy struct from GPU to CPU to get pointer addresses
    GpuOrderedValuesData h_gpu_data;
    CUDA_CHECK(cudaMemcpy(&h_gpu_data, d_gpu_data, sizeof(GpuOrderedValuesData), cudaMemcpyDeviceToHost));
    
    // Release each array on GPU
    if (h_gpu_data.letter_all_values) {
        CUDA_CHECK(cudaFree(h_gpu_data.letter_all_values));
        h_gpu_data.letter_all_values = nullptr;
    }
    if (h_gpu_data.digit_all_values) {
        CUDA_CHECK(cudaFree(h_gpu_data.digit_all_values));
        h_gpu_data.digit_all_values = nullptr;
    }
    if (h_gpu_data.symbol_all_values) {
        CUDA_CHECK(cudaFree(h_gpu_data.symbol_all_values));
        h_gpu_data.symbol_all_values = nullptr;
    }
    
    if (h_gpu_data.letter_value_offsets) {
        CUDA_CHECK(cudaFree(h_gpu_data.letter_value_offsets));
        h_gpu_data.letter_value_offsets = nullptr;
    }
    if (h_gpu_data.digits_value_offsets) {
        CUDA_CHECK(cudaFree(h_gpu_data.digits_value_offsets));
        h_gpu_data.digits_value_offsets = nullptr;
    }
    if (h_gpu_data.symbol_value_offsets) {
        CUDA_CHECK(cudaFree(h_gpu_data.symbol_value_offsets));
        h_gpu_data.symbol_value_offsets = nullptr;
    }
    
    if (h_gpu_data.letter_seg_offsets) {
        CUDA_CHECK(cudaFree(h_gpu_data.letter_seg_offsets));
        h_gpu_data.letter_seg_offsets = nullptr;
    }
    if (h_gpu_data.digit_seg_offsets) {
        CUDA_CHECK(cudaFree(h_gpu_data.digit_seg_offsets));
        h_gpu_data.digit_seg_offsets = nullptr;
    }
    if (h_gpu_data.symbol_seg_offsets) {
        CUDA_CHECK(cudaFree(h_gpu_data.symbol_seg_offsets));
        h_gpu_data.symbol_seg_offsets = nullptr;
    }
    
    // Release the struct itself
    CUDA_CHECK(cudaFree(d_gpu_data));
    d_gpu_data = nullptr;

}

// Function to clean global variables
void cleanup_global_cuda_resources() {
    if (gpu_data != nullptr) {
        clean_gpu_ordered_values_data(gpu_data);
        gpu_data = nullptr;
    }
    
    if (task_manager != nullptr) {
        // Just delete directly, STL will release itself.
        // task_manager->clean();
        delete task_manager;
        task_manager = nullptr;
    }
    
}





