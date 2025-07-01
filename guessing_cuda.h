#ifndef GUESSING_CUDA_H
#define GUESSING_CUDA_H

#include "PCFG.h"
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
extern GpuOrderedValuesData* gpu_data;


// 声明CUDA函数，这样.cpp文件就能调用它们
void init_gpu_ordered_values_data(
    GpuOrderedValuesData* d_gpu_data,
    PriorityQueue& q
);

// 如果有其他CUDA函数也在此声明

struct Taskcontent{
    int segment_type; // 1: letter, 2: digit, 3: symbol
    int seg_id;
    char* prefix;
    int prefix_length; // prefix的长度
    
};

class TaskManager{

};

__global__ generate_guesses_kernel(
    GpuOrderedValuesData* gpu_data,
    Taskcontent* tasks
)





#endif