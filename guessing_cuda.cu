#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <string_view>
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
char* h_guess_buffer = nullptr; // 为了方便在别的地方释放。。。。在用 string_view

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


void TaskManager::add_task(segment* seg, string prefix, PriorityQueue& q) {

#ifdef TIME_COUNT
auto start_add_task = system_clock::now();
#endif
    // 确保映射表已初始化
    if (!maps_initialized) {
        init_length_maps(q);
    }
    
    seg_types.push_back(seg->type);
    seg_lens.push_back(seg->length);
    
    switch (seg->type) {
    case 1:
        seg_ids.push_back(letter_length_to_id[seg->length]);
        break;
    case 2:
        seg_ids.push_back(digit_length_to_id[seg->length]);
        break;
    case 3:
        seg_ids.push_back(symbol_length_to_id[seg->length]);
        break;
    default:
        throw "undefined_segment_error";
        break;
    }
    
    prefixs.push_back(prefix);
    prefix_lens.push_back(prefix.length());
    taskcount++;
    seg_value_count.push_back(seg->ordered_values.size());  
    guesscount += seg->ordered_values.size();

#ifdef DEBUG
    cout << "Added task: ";
    seg->PrintSeg();
    cout << " -> ID: " << seg_ids.back() << endl;
#endif


#ifdef TIME_COUNT
auto end_add_task = system_clock::now();
auto duration_add_task = duration_cast<microseconds>(end_add_task - start_add_task);
time_add_task += double(duration_add_task.count()) * microseconds::period::num / microseconds::period::den;
#endif

}


void TaskManager::init_length_maps(PriorityQueue& q) {
    if (maps_initialized) return;
    
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
    
    maps_initialized = true;
    
#ifdef DEBUG
    cout << "初始化长度映射完成:" << endl;
    cout << "  字母长度映射: " << letter_length_to_id.size() << " 种长度" << endl;
    cout << "  数字长度映射: " << digit_length_to_id.size() << " 种长度" << endl;
    cout << "  符号长度映射: " << symbol_length_to_id.size() << " 种长度" << endl;
#endif
}

void TaskManager::launch_gpu_kernel(vector<string_view>& guesses, PriorityQueue& q){
#ifdef TIME_COUNT
auto start_one_batch = system_clock::now();


auto start_before_launch = system_clock::now();
#endif

    //1. 准备数据
    Taskcontent h_tasks; 
    Taskcontent temp; // 为了中转一下 gpu的地址...(h_tasks里是cpu的)
    
    Taskcontent* d_tasks;
    char* d_guess_buffer;
    size_t result_len = 0; // 结果数组的长度
    vector<int> res_offset;// 结果char* 中， 每个seg对应的一坨guess的开始offset。 

#ifdef DEBUG
    cout << "=== TaskManager::launch_gpu_kernel 调试信息 ===" << endl;
    cout << "任务数量: " << taskcount << endl;
    cout << "预计guess数量: " << guesscount << endl;
    cout << "Segment信息:" << endl;
    for (int i = 0; i < taskcount; i++) {
        cout << "  Task " << i << ": type=" << seg_types[i] 
             << ", id=" << seg_ids[i] 
             << ", len=" << seg_lens[i] 
             << ", value_count=" << seg_value_count[i] 
             << ", prefix='" << prefixs[i] << "' (len=" << prefix_lens[i] << ")" << endl;
    }
#endif


    h_tasks.seg_types = seg_types.data();
    h_tasks.seg_ids = seg_ids.data();
    string all_prefixes = std::accumulate(prefixs.begin(), prefixs.end(), std::string(""));


#ifdef DEBUG
    cout << "\n连接后的prefixes: '" << all_prefixes << "'" << endl;
    cout << "连接后的prefixes长度: " << all_prefixes.length() << endl;
#endif


    h_tasks.prefixs = all_prefixes.c_str();
    h_tasks.prefix_offsets = new int[prefixs.size() + 1]; // +1 for the end offset
    h_tasks.prefix_offsets[0] = 0; // 第一个prefix的起始位置是0
    for (size_t i = 0; i < prefixs.size(); ++i) {
        h_tasks.prefix_offsets[i + 1] = h_tasks.prefix_offsets[i] + prefix_lens[i]; // 计算每个prefix的起始位置
    }


#ifdef DEBUG
    cout << "\nPrefix偏移信息:" << endl;
    for (size_t i = 0; i <= prefixs.size(); i++) {
        cout << "  prefix_offsets[" << i << "] = " << h_tasks.prefix_offsets[i] << endl;
    }
    
    cout << "\n验证prefix提取:" << endl;
    for (size_t i = 0; i < prefixs.size(); i++) {
        int start = h_tasks.prefix_offsets[i];
        int len = prefix_lens[i];
        string extracted_prefix(all_prefixes.substr(start, len));
        cout << "  Task " << i << ": 原始='" << prefixs[i] 
             << "', 提取='" << extracted_prefix << "'" 
             << " " << (prefixs[i] == extracted_prefix ? "✓" : "✗") << endl;
    }
#endif

    h_tasks.prefix_lens = prefix_lens.data();
    h_tasks.taskcount = taskcount;
    h_tasks.guesscount = guesscount;
    h_tasks.seg_lens  = seg_lens.data();
    h_tasks.seg_value_counts = seg_value_count.data(); // 每个segment的value数量
    
    // 合并计算：同时得到 gpu_buffer 数组的长度和累积guess偏移数组
    vector<int> cumulative_offsets(taskcount + 1, 0);
    for(int i = 0; i < taskcount; i++){
        res_offset.push_back(result_len);
        result_len += seg_value_count[i]*(seg_lens[i] + prefix_lens[i]);
        cumulative_offsets[i + 1] = cumulative_offsets[i] + seg_value_count[i];
    }
    h_tasks.output_offsets = res_offset.data(); // 这样的话，就没有存最末尾的。只有taskcount个
    h_tasks.cumulative_guess_offsets = cumulative_offsets.data();


#ifdef DEBUG
    cout << "\n结果缓冲区计算:" << endl;
    cout << "总结果长度: " << result_len << " 字符" << endl;
    cout << "每个Task的偏移和长度:" << endl;
    for(int i = 0; i < seg_value_count.size(); i++){
        int task_total_len = seg_value_count[i] * (seg_lens[i] + prefix_lens[i]);
        cout << "  Task " << i << ": offset=" << res_offset[i] 
             << ", 单个guess长度=" << (seg_lens[i] + prefix_lens[i])
             << ", guess数量=" << seg_value_count[i]
             << ", 总长度=" << task_total_len << endl;
    }
    
    // 验证总长度计算
    int calculated_total = 0;
    for(int i = 0; i < seg_value_count.size(); i++){
        calculated_total += seg_value_count[i] * (seg_lens[i] + prefix_lens[i]);
    }
    cout << "验证总长度: " << calculated_total << " (应该等于 " << result_len << ") " 
         << (calculated_total == result_len ? "✓" : "✗") << endl;
    cout << "\n累积偏移数组:" << endl;
    for(int i = 0; i <= taskcount; i++){
        cout << "  cumulative_offsets[" << i << "] = " << cumulative_offsets[i] << endl;
    }

#endif


    // 1.999. 分配cpu 部分的内存 （结果 buffer ）
    h_guess_buffer = new char[result_len]; // host端的guess_buffer (结果 buffer)


#ifdef DEBUG
    // 检查内存分配
    if (h_guess_buffer == nullptr) {
        cout << "错误: 无法分配 " << result_len << " 字节的内存!" << endl;
        return;
    } else {
        cout << "成功分配 " << result_len << " 字节的主机内存" << endl;
    }
    
    // 验证数据一致性
    cout << "\n数据一致性检查:" << endl;
    int total_guesses_check = 0;
    for (int i = 0; i < seg_value_count.size(); i++) {
        total_guesses_check += seg_value_count[i];
    }
    cout << "计算得出的总guess数: " << total_guesses_check 
         << " (应该等于 " << guesscount << ") " 
         << (total_guesses_check == guesscount ? "✓" : "✗") << endl;
    
    if (seg_types.size() != taskcount || 
        seg_ids.size() != taskcount || 
        seg_lens.size() != taskcount || 
        prefixs.size() != taskcount || 
        prefix_lens.size() != taskcount || 
        seg_value_count.size() != taskcount) {
        cout << "警告: 数组大小不一致!" << endl;
        cout << "  seg_types.size() = " << seg_types.size() << endl;
        cout << "  seg_ids.size() = " << seg_ids.size() << endl;
        cout << "  seg_lens.size() = " << seg_lens.size() << endl;
        cout << "  prefixs.size() = " << prefixs.size() << endl;
        cout << "  prefix_lens.size() = " << prefix_lens.size() << endl;
        cout << "  seg_value_count.size() = " << seg_value_count.size() << endl;
        cout << "  taskcount = " << taskcount << endl;
    } else {
        cout << "所有数组大小一致 ✓" << endl;
    }
    
    cout << "=== 调试信息结束 ===" << endl << endl;
#endif

    //2. 分配gpu 内存 以及 
    //3.copy
    // mem_allocate_and_copy(tasks);
    //分配gpu内存
    char* temp_prefixs;
    CUDA_CHECK(cudaMalloc(&temp_prefixs, h_tasks.prefix_offsets[prefixs.size()] * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&temp.seg_types, taskcount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&temp.seg_ids, taskcount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&temp.seg_lens, seg_lens.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&temp.prefix_offsets, (prefixs.size() + 1 ) * sizeof(int))); // +1 for the end offset
    CUDA_CHECK(cudaMalloc(&temp.prefix_lens, prefix_lens.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&temp.seg_value_counts, seg_value_count.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&temp.cumulative_guess_offsets, (taskcount + 1) * sizeof(int))); // 累积偏移数组
    CUDA_CHECK(cudaMalloc(&temp.output_offsets, (taskcount + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tasks, sizeof(Taskcontent)));


    // 分配结果guess_buffer (gpu上)
    CUDA_CHECK(cudaMalloc(&d_guess_buffer, result_len * sizeof(char)));

    //进行copy
    CUDA_CHECK(cudaMemcpy(temp.seg_types, h_tasks.seg_types, taskcount * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(temp.seg_ids, h_tasks.seg_ids, taskcount * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(temp.seg_lens, h_tasks.seg_lens, seg_lens.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(temp_prefixs, h_tasks.prefixs,  h_tasks.prefix_offsets[prefixs.size()] * sizeof(char), cudaMemcpyHostToDevice));
    temp.prefixs = temp_prefixs; // 直接指向gpu的地址
    
    // 最后一个偏移量。。。指的是prefixs的结尾 + 1 的下标，也就是总长度
    CUDA_CHECK(cudaMemcpy(temp.prefix_offsets, h_tasks.prefix_offsets, (prefixs.size() + 1 ) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(temp.prefix_lens, h_tasks.prefix_lens, prefixs.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(temp.seg_value_counts, h_tasks.seg_value_counts, seg_value_count.size() *sizeof(int), cudaMemcpyHostToDevice)); 
    CUDA_CHECK(cudaMemcpy(temp.cumulative_guess_offsets, h_tasks.cumulative_guess_offsets, (taskcount + 1) * sizeof(int), cudaMemcpyHostToDevice)); // 复制累积偏移数组
    CUDA_CHECK(cudaMemcpy(temp.output_offsets, h_tasks.output_offsets,  (taskcount + 1) * sizeof(int), cudaMemcpyHostToDevice)); 


    //int 直接赋值到 temp里 再copy到gpu ，应该可以
    temp.taskcount = taskcount;
    temp.guesscount = guesscount;

    CUDA_CHECK(cudaMemcpy(d_tasks, &temp, sizeof(Taskcontent), cudaMemcpyHostToDevice));
    
#ifdef TIME_COUNT
auto end_before_launch = system_clock::now();
auto duration_before_launch = duration_cast<microseconds>(end_before_launch - start_before_launch);
time_before_launch += double(duration_before_launch.count()) * microseconds::period::num / microseconds::period::den;



#endif
    

    //4. 启动kernal 开始计算
    //TODO 看一下到底能启用多少thread？ 这里不很清楚该怎么处理
    int total_threads_needed = (guesscount + GUESS_PER_THREAD - 1) / GUESS_PER_THREAD;
    int threads_per_block = 1024;
    int blocks = (total_threads_needed + threads_per_block - 1) / threads_per_block;
    
//TODO ，看看每次启用多少线程， 以及尝试二分查找来找 guess对应的task
#ifdef DEBUG  
    cout << "启动kernel: blocks=" << blocks << ", threads_per_block=" << threads_per_block << endl;
    cout << "总线程数: " << blocks * threads_per_block << ", guess数量: " << guesscount << endl;
#endif


#ifdef TIME_COUNT
auto start_launch = system_clock::now();
#endif
    generate_guesses_kernel<<<blocks, threads_per_block>>>(gpu_data, d_tasks, d_guess_buffer);
    
    // 检查kernel启动错误
    CUDA_CHECK(cudaGetLastError());



    //5. 从GPU获取结果
    CUDA_CHECK(cudaDeviceSynchronize());// 等待gpu 完成计算

#ifdef TIME_COUNT
auto end_launch = system_clock::now();
auto duration_launch = duration_cast<microseconds>(end_launch - start_launch);
time_launch_task += double(duration_launch.count()) * microseconds::period::num / microseconds::period::den;

// 完成计算到填好
auto start_after_launch = system_clock::now();
#endif

#ifdef TIME_COUNT
auto start_memcpy_toh = system_clock::now();
#endif
    CUDA_CHECK(cudaMemcpy(h_guess_buffer, d_guess_buffer, result_len * sizeof(char), cudaMemcpyDeviceToHost));

#ifdef TIME_COUNT
auto end_memcpy_toh = system_clock::now();
auto duration_memcpy_toh = duration_cast<microseconds>(end_memcpy_toh - start_memcpy_toh);
time_memcpy_toh += double(duration_memcpy_toh.count()) * microseconds::period::num / microseconds::period::den;
#endif

    //6. 将结果填入guesses
#ifdef TIME_COUNT
auto start_string_process = system_clock::now();
#endif
    //BUG 由于想用string_view ，然后char*指针想在外面释放，所以为了简便，每次生成完都要hash然后把
    // 相应指针释放了， 不然会出事。 所以每个gpu batch 的长度要大于100000
    // 由于每个 seg 对应的 guess 们的长度是一样的， 所以这么搞
    // for(int i = 0; i < seg_ids.size(); i++) {
    //     for(int j = 0; j < seg_value_count[i]; j++) {
    //         int start_offset = res_offset[i] + j*(seg_lens[i] + prefix_lens[i]);
    //         string guess(h_guess_buffer + start_offset, h_guess_buffer + start_offset + seg_lens[i] + prefix_lens[i]);
    //         guesses.push_back(guess);
    //     }
    // }

    guesses.reserve(guesscount);
    for (int i = 0; i < seg_ids.size(); i++) {
        for (int j = 0; j < seg_value_count[i]; j++) {
            int start_offset = res_offset[i] + j * (seg_lens[i] + prefix_lens[i]);
            std::string_view guess(
                h_guess_buffer + start_offset,
                seg_lens[i] + prefix_lens[i]
            );
            guesses.emplace_back(
                h_guess_buffer + start_offset,
                seg_lens[i] + prefix_lens[i]
            );
        }
    }


#ifdef TIME_COUNT
auto end_string_process = system_clock::now();
auto duration_string_process = duration_cast<microseconds>(end_string_process - start_string_process);
time_string_process += double(duration_string_process.count()) * microseconds::period::num / microseconds::period::den;
#endif



    //7. 释放内存
#ifdef DEBUG
    cout << "开始释放GPU内存..." << endl;
#endif

    // 释放GPU内存
    CUDA_CHECK(cudaFree(temp_prefixs));
    CUDA_CHECK(cudaFree(temp.seg_types));
    CUDA_CHECK(cudaFree(temp.seg_ids));
    CUDA_CHECK(cudaFree(temp.seg_lens));
    CUDA_CHECK(cudaFree(temp.prefix_offsets));
    CUDA_CHECK(cudaFree(temp.prefix_lens));
    CUDA_CHECK(cudaFree(temp.seg_value_counts));
    CUDA_CHECK(cudaFree(temp.cumulative_guess_offsets)); // 释放累积偏移数组
    CUDA_CHECK(cudaFree(temp.output_offsets));
    CUDA_CHECK(cudaFree(d_tasks));
    CUDA_CHECK(cudaFree(d_guess_buffer));
    
    // 将指针置空以避免悬空指针
    temp_prefixs = nullptr;
    temp.seg_types = nullptr;
    temp.seg_ids = nullptr;
    temp.seg_lens = nullptr;
    temp.prefix_offsets = nullptr;
    temp.prefix_lens = nullptr;
    temp.seg_value_counts = nullptr;
    temp.cumulative_guess_offsets = nullptr; // 置空累积偏移数组指针
    temp.prefixs = nullptr;
    temp.output_offsets = nullptr;
    d_tasks = nullptr;
    d_guess_buffer = nullptr;
    
    // 释放CPU内存
    // delete[] h_guess_buffer;
    delete[] h_tasks.prefix_offsets;
    // h_guess_buffer = nullptr;
    h_tasks.prefix_offsets = nullptr;

#ifdef DEBUG
    cout << "GPU内存释放完成" << endl;
#endif


    //8. 清理TaskManager
    clean();

#ifdef TIME_COUNT
auto end_after_launch = system_clock::now();
auto duration_after_launch = duration_cast<microseconds>(end_after_launch - start_after_launch);
time_after_launch += double(duration_after_launch.count()) * microseconds::period::num / microseconds::period::den;

auto end_one_batch = system_clock::now();
auto duration_one_batch = duration_cast<microseconds>(end_one_batch - start_one_batch);
time_all_batch += double(duration_one_batch.count()) * microseconds::period::num / microseconds::period::den;
#endif
}


void TaskManager::clean() {
    // 清理TaskManager中的数据
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

#ifdef DEBUG
    cout << "start init gpu" <<endl;
#endif
    //cpu上的数据
    GpuOrderedValuesData h_gpu_data;

    // 计算char*数组总长度
    size_t total_letter_length = 0;
    size_t total_digit_length = 0;
    size_t total_symbol_length = 0;

    // 有多少个各类型的 value 
    size_t letter_offsetarr_length = 0;
    size_t digit_offsetarr_length = 0;
    size_t symbol_offsetarr_length = 0;

    char* letter_all_values = nullptr;// 把各个segment 的ordered_values展平
    char* digit_all_values= nullptr;
    char* symbol_all_values= nullptr;

    int* letter_value_offsets= nullptr; // 每个ordered_value在letter_all_values中的起始
    int* digits_value_offsets= nullptr; // 每个ordered_value在digit_all_values中的起始位置
    int* symbol_value_offsets= nullptr; // 每个ordered_value在symbol_all_values中的起始

    int* letter_seg_offsets= nullptr; // 每个letter segment第一个ordered_value在value_offsets中的是哪
    int* digit_seg_offsets= nullptr; // 每个digit segment第一个ordered_value在value_offsets中的是哪个
    int* symbol_seg_offsets= nullptr; // 每个symbol segment第一个ordered_value在value_offsets
#ifdef DEBUG
    cout <<"end init local var" <<endl;
#endif
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
#ifdef DEBUG
    cout <<"part 1" <<endl;
#endif

    letter_all_values = new char[total_letter_length];
    letter_value_offsets = new int[letter_offsetarr_length+1];
    letter_seg_offsets = new int[q.m.letters.size()+1];

    digit_all_values = new char[total_digit_length];
    digits_value_offsets = new int[digit_offsetarr_length+1];
    digit_seg_offsets = new int[q.m.digits.size()+1];

    symbol_all_values = new char[total_symbol_length];
    symbol_value_offsets = new int[symbol_offsetarr_length+1];
    symbol_seg_offsets = new int[q.m.symbols.size()+1]; 

#ifdef DEBUG
    cout <<"part 1.1" <<endl;
#endif
    // 多补了一个offset 来表示末尾，实际不对应segment 以及 value

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
#ifdef DEBUG
    cout <<"part 1.2"<<endl;
#endif
    letter_seg_offsets[q.m.letters.size()] = seg_offset; // 最后补一下...其实对应长度为0
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
#ifdef DEBUG
    cout <<"part 1.3"<<endl;
#endif
    digit_seg_offsets[q.m.digits.size()] = seg_offset; // 最后补一下...其实对应长度为0
    digits_value_offsets[digit_offsetarr_length] = value_offset;
    seg_offset = 0;
    value_offset = 0;   
#ifdef DEBUG
    cout <<"part 2" <<endl;
#endif
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
    symbol_seg_offsets[q.m.symbols.size()] = seg_offset; // 最后补一下
    symbol_value_offsets[symbol_offsetarr_length] = value_offset;

#ifdef DEBUG
    cout <<"part 3" <<endl;
#endif
    // 相关东西都要的是地址。。。。。。。 指针只是指针， 解释成cpu or gpu 的内存 ，是看具体情景
    // 比如cudaMemcpy 用cudaMemcpyKind kind 来区分。

    //把各个指针相应数据复制到gpu上
#ifdef DEBUG
    printf("total_letter_length: %zu\n", total_letter_length);
    printf("total_digit_length: %zu\n", total_digit_length);
    printf("total_symbol_length: %zu\n", total_symbol_length);
    printf("letter_offsetarr_length: %zu\n", letter_offsetarr_length);
    printf("digit_offsetarr_length: %zu\n", digit_offsetarr_length);
    printf("symbol_offsetarr_length: %zu\n", symbol_offsetarr_length);

    //TODO 也许可以打印一下 q.m. 里面的那些长度看看有没有问题。

    // 看看letter 的前十个是否有问题。
    for(int i=0;i<10;i++){
        segment& seg = q.m.letters[i];  
        seg.PrintSeg();
        seg.PrintValues();
        // for(int j = 0; j < seg.ordered_values.size(); j++) {
        //     string letter_value(h_gpu_data.letter_all_values + 
        //         h_gpu_data.letter_value_offsets[h_gpu_data.letter_seg_offsets[i]] + 
        //         j * seg.length, seg.length );
        //     cout << letter_value << " ";
        // }
        cout << endl <<endl;
    }

    // 另外俩也可以以类似方式看是否有问题。
#endif

#ifdef DEBUG
    cout <<"part 4" <<endl;
#endif

    // 分配内存 以及复制
    CUDA_CHECK(cudaMalloc(&h_gpu_data.letter_all_values, total_letter_length * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&h_gpu_data.digit_all_values, total_digit_length * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&h_gpu_data.symbol_all_values, total_symbol_length * sizeof(char)));
    CUDA_CHECK(cudaMemcpy(h_gpu_data.letter_all_values, letter_all_values, total_letter_length * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_gpu_data.digit_all_values, digit_all_values, total_digit_length * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_gpu_data.symbol_all_values, symbol_all_values, total_symbol_length * sizeof(char), cudaMemcpyHostToDevice));

    // 分配偏移数组 以及复制
    CUDA_CHECK(cudaMalloc(&h_gpu_data.letter_value_offsets, (letter_offsetarr_length+1)* sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h_gpu_data.digits_value_offsets, (digit_offsetarr_length+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h_gpu_data.symbol_value_offsets, (symbol_offsetarr_length+1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(h_gpu_data.letter_value_offsets, letter_value_offsets, (letter_offsetarr_length+1)* sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_gpu_data.digits_value_offsets, digits_value_offsets, (digit_offsetarr_length+1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_gpu_data.symbol_value_offsets, symbol_value_offsets, (symbol_offsetarr_length+1) * sizeof(int), cudaMemcpyHostToDevice));


    // 分配segment偏移数组 以及复制
    CUDA_CHECK(cudaMalloc(&h_gpu_data.letter_seg_offsets, q.m.letters.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h_gpu_data.digit_seg_offsets, q.m.digits.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h_gpu_data.symbol_seg_offsets, q.m.symbols.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(h_gpu_data.letter_seg_offsets, letter_seg_offsets, q.m.letters.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_gpu_data.digit_seg_offsets, digit_seg_offsets, q.m.digits.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_gpu_data.symbol_seg_offsets, symbol_seg_offsets, q.m.symbols.size() * sizeof(int), cudaMemcpyHostToDevice));


    //把结构体复制到gpu上
    CUDA_CHECK(cudaMalloc(&d_gpu_data, sizeof(GpuOrderedValuesData)));
    CUDA_CHECK(cudaMemcpy(d_gpu_data, &h_gpu_data, sizeof(GpuOrderedValuesData), cudaMemcpyHostToDevice));

    // 释放CPU端的临时内存
    delete[] letter_all_values;
    delete[] digit_all_values;
    delete[] symbol_all_values;
    delete[] letter_value_offsets;
    delete[] digits_value_offsets;
    delete[] symbol_value_offsets;
    delete[] letter_seg_offsets;
    delete[] digit_seg_offsets;
    delete[] symbol_seg_offsets;

#ifdef DEBUG
    cout << "GPU数据初始化完成，CPU临时内存已释放" << endl;
#endif

}

void clean_gpu_ordered_values_data(GpuOrderedValuesData *& d_gpu_data){
    if (d_gpu_data == nullptr) return;
    
#ifdef DEBUG
    cout << "清理GPU ordered values 数据..." << endl;
#endif

    // 从GPU复制结构体到CPU以获取指针地址
    GpuOrderedValuesData h_gpu_data;
    CUDA_CHECK(cudaMemcpy(&h_gpu_data, d_gpu_data, sizeof(GpuOrderedValuesData), cudaMemcpyDeviceToHost));
    
    // 释放GPU上的各个数组
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
    
    // 释放结构体本身
    CUDA_CHECK(cudaFree(d_gpu_data));
    d_gpu_data = nullptr;

#ifdef DEBUG
    cout << "GPU ordered values 数据清理完成" << endl;
#endif
}

// 清理全局变量的函数
void cleanup_global_cuda_resources() {
    if (gpu_data != nullptr) {
        clean_gpu_ordered_values_data(gpu_data);
        gpu_data = nullptr;
    }
    
    if (task_manager != nullptr) {
        // 直接删就好了，STL自己会释放。
        // task_manager->clean();
        delete task_manager;
        task_manager = nullptr;
    }
    
#ifdef DEBUG
    cout << "全局CUDA资源清理完成" << endl;
#endif
}





