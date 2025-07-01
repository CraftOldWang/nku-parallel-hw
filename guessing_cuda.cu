#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include "guessing_cuda.h"  // 包含头文件
using namespace std;

// vector<segment> letters;
// vector<segment> digits;
// vector<segment> symbols;
// 把所有segment的ordered_values放在一个全局变量中
// vector<string> ordered_values;
// 偏移，某个指针
// 某一个int 数字， 对应
// 只需要 给一个seg->给出对应的下标 , 这个下标（int的） 直接在gpu 那里也能用
GpuOrderedValuesData* gpu_data = nullptr;


__global__ void generate_guesses_kernel(
    GpuOrderedValuesData* gpu_data,
    int* segment_offsets,
    int segment_count,
    int* output_guesses,
    int* output_count
) {
    // 这里实现具体的kernel逻辑
    // 需要根据gpu_data中的数据生成相应的guesses
    // 注意：这里只是一个示例，具体实现需要根据实际需求来编写
}


void init_gpu_ordered_values_data(
    GpuOrderedValuesData* d_gpu_data,
    PriorityQueue& q
) {

    // 计算char*数组总长度
    size_t total_letter_length = 0;
    size_t total_digit_length = 0;
    size_t total_symbol_length = 0;

    // 有多少个不同类型的 value
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
    letter_seg_offsets[q.m.letters.size()] = seg_offset; // 最后补一下...其实对应长度为0
    letter_value_offsets[letter_offsetarr_length] = value_offset;
    seg_offset = 0;
    value_offset = 0;

    for (int i=0;i< q.m.digits.size();i++) {
        digit_seg_offsets[i] = seg_offset;
        const auto& seg = q.m.digits[i];
        seg_offset += seg.ordered_values.size();
        for (int j=0; j< seg.ordered_values.size();j++ ) {
            digits_value_offsets[seg_offset++] = value_offset;
            string value = seg.ordered_values[j];
            for(int k = 0;k < value.length(); k++) {
                digit_all_values[value_offset++] = value[k];
            }
        }
    }
    digit_seg_offsets[q.m.digits.size()] = seg_offset; // 最后补一下...其实对应长度为0
    digits_value_offsets[digit_offsetarr_length] = value_offset;
    seg_offset = 0;
    value_offset = 0;   


    for (int i=0;i< q.m.symbols.size();i++) {
        symbol_seg_offsets[i] = seg_offset;
        const auto& seg = q.m.symbols[i];
        seg_offset += seg.ordered_values.size();
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



    // 相关东西都要的是地址。。。。。。。 指针只是指针， 解释成cpu or gpu 的内存 ，是看具体情景
    // 比如cudaMemcpy 用cudaMemcpyKind kind 来区分。

    GpuOrderedValuesData h_gpu_data;
    //把各个指针相应数据复制到gpu上

    // 分配内存 以及复制
    cudaMalloc(&h_gpu_data.letter_all_values, total_letter_length * sizeof(char));
    cudaMalloc(&h_gpu_data.digit_all_values, total_digit_length * sizeof(char));
    cudaMalloc(&h_gpu_data.symbol_all_values, total_symbol_length * sizeof(char));
    cudaMemcpy(h_gpu_data.letter_all_values, letter_all_values, total_letter_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(h_gpu_data.digit_all_values, digit_all_values, total_digit_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(h_gpu_data.symbol_all_values, symbol_all_values, total_symbol_length * sizeof(char), cudaMemcpyHostToDevice);

    // 分配偏移数组 以及复制
    cudaMalloc(&h_gpu_data.letter_value_offsets, (letter_offsetarr_length+1)* sizeof(int));
    cudaMalloc(&h_gpu_data.digits_value_offsets, (digit_offsetarr_length+1) * sizeof(int));
    cudaMalloc(&h_gpu_data.symbol_value_offsets, (symbol_offsetarr_length+1) * sizeof(int));
    cudaMemcpy(h_gpu_data.letter_value_offsets, letter_value_offsets, (letter_offsetarr_length+1)* sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(h_gpu_data.digits_value_offsets, digits_value_offsets, (digit_offsetarr_length+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(h_gpu_data.symbol_value_offsets, symbol_value_offsets, (symbol_offsetarr_length+1) * sizeof(int), cudaMemcpyHostToDevice);


    // 分配segment偏移数组 以及复制
    cudaMalloc(&h_gpu_data.letter_seg_offsets, q.m.letters.size() * sizeof(int));
    cudaMalloc(&h_gpu_data.digit_seg_offsets, q.m.digits.size() * sizeof(int));
    cudaMalloc(&h_gpu_data.symbol_seg_offsets, q.m.symbols.size() * sizeof(int));
    cudaMemcpy(h_gpu_data.letter_seg_offsets, letter_seg_offsets, q.m.letters.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(h_gpu_data.digit_seg_offsets, digit_seg_offsets, q.m.digits.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(h_gpu_data.symbol_seg_offsets, symbol_seg_offsets, q.m.symbols.size() * sizeof(int), cudaMemcpyHostToDevice);


    //把结构体复制到gpu上
    cudaMalloc(&d_gpu_data, sizeof(GpuOrderedValuesData));
    cudaMemcpy(d_gpu_data, &h_gpu_data, sizeof(GpuOrderedValuesData), cudaMemcpyHostToDevice);

}



