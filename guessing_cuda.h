#ifndef GUESSING_CUDA_H
#define GUESSING_CUDA_H
#include <vector>
#include "PCFG.h"
#include <numeric> // for std::accumulate
#include <device_launch_parameters.h>

#include "config.h"

// GPU上用于查找的 ordered_values 数据结构
struct GpuOrderedValuesData{
    char* letter_all_values;// 把各个segment 的ordered_values展平
    char* digit_all_values;
    char* symbol_all_values;

    int* letter_value_offsets; // 每个ordered_value在letter_all_values中的起始
    int* digits_value_offsets; // 每个ordered_value在digit_all_values中的起始位置
    int* symbol_value_offsets; // 每个ordered_value在symbol_all_values中的起始

    int* letter_seg_offsets; // 每个letter segment第一个ordered_value在value_offsets中的是哪个
    int* digit_seg_offsets; // 每个digit segment第一个ordered_value在value_offsets中的是哪个
    int* symbol_seg_offsets; // 每个symbol segment第一个ordered_value在value_offsets
};


// 给GPU用的 一批任务 。包含了生成guess所需的数据
struct Taskcontent{ 
    int* seg_types; // 1: letter, 2: digit, 3: symbol (0:未设置)
    int* seg_ids;   // 对应在 model 里的 三个vector 的下标
    int* seg_lens;
    const char* prefixs;
    int* prefix_offsets;
    int* prefix_lens; // 每个prefix的长度
    int* seg_value_counts; // 每个segment的value数量
    int* cumulative_guess_offsets; // 累积guess偏移数组，用于二分查找优化

    int* output_offsets;  // 每个task在输出buffer中的起始位置

    int taskcount;
    int guesscount; // 到10_0000了就会丢给核函数去执行
};

class TaskManager{
public:
    vector<int> seg_types;
    vector<int> seg_ids;
    vector<int> seg_lens;
    vector<string> prefixs;
    vector<int> prefix_lens; // 每个prefix的长度

    vector<int> seg_value_count; // 每个seg 有多少 value （其实只会有所谓最后一个seg的， 就是后面一个seg生成相应数量guess）
    int taskcount;
    int guesscount; // 到10_0000了就会丢给核函数去执行

    TaskManager(): taskcount(0), guesscount(0){}// vector 自己会调自己的构造..
    void add_task(segment* seg, string prefix, PriorityQueue& q);
    void launch_gpu_kernel(vector<string_view>& guesses, PriorityQueue& q);
    void clean();
    void print();
    void init_length_maps(PriorityQueue& q);

private:
    // 添加长度到ID的映射表
    unordered_map<int, int> letter_length_to_id;
    unordered_map<int, int> digit_length_to_id;
    unordered_map<int, int> symbol_length_to_id;
    bool maps_initialized = false;
};

// 最后没用上
// 用二分查找找 taskid 的函数（不一定会用上）
// __device__ int find_task_id(int guess_id, int* cumulative_offsets, int task_count) {
//     int left = 0, right = task_count - 1;
//     while (left <= right) {
//         int mid = (left + right) / 2;
//         if (guess_id >= cumulative_offsets[mid] && guess_id < cumulative_offsets[mid + 1]) {
//             return mid;
//         } else if (guess_id < cumulative_offsets[mid]) {
//             right = mid - 1;  
//         } else {
//             left = mid + 1;
//         }
//     }
//     return task_count - 1; // 安全返回
// }



// 生成猜测的 kernal 函数 。生成的猜测放到 d_guess_buffer 上
__global__ void generate_guesses_kernel(
    GpuOrderedValuesData* gpu_data,
    Taskcontent* d_tasks,
    char* d_guess_buffer
);

// 全局变量声明
extern GpuOrderedValuesData* gpu_data;
extern TaskManager* task_manager;

// 初始化函数、清理函数声明

// 初始化gpu上用于查找的数据 (segment表)
void init_gpu_ordered_values_data(GpuOrderedValuesData*& d_gpu_data, PriorityQueue& q);
// 清理gpu上用来查找的数据 (segment表)
void clean_gpu_ordered_values_data(GpuOrderedValuesData*& d_gpu_data);
// 清理 segment表 和 taskmanager (清理两个全局变量) 
void cleanup_global_cuda_resources();

#endif