#ifndef GUESSING_CUDA_H
#define GUESSING_CUDA_H

#include "PCFG.h"
#include <cuda_runtime.h>
#include <vector>
#include <string>

// 使用前向声明以避免循环依赖
class CUDABatchManager;

// 描述一个独立的生成任务
struct GuessTask {
    // 任务来源的PT信息
    const PT* source_pt;
    
    // 任务的前缀（对于多段PT）
    std::string prefix;
    
    // 任务需要使用的值列表（指向segment中的ordered_values）
    const std::vector<std::string>* values;
    
    // 这个任务将生成多少个猜测
    int num_guesses_to_generate;
};

// 统一的CUDA核函数，用于批处理
__global__ void generateGuessesKernel_Batch(
    const GuessTask* d_tasks,
    int num_tasks,
    char* d_result_buffer,
    int* d_result_lengths,
    // ... 其他必要的扁平化数据指针
);

// 新的管理器类，用于处理CUDA批处理
class CUDABatchManager {
public:
    CUDABatchManager(PriorityQueue& pq);
    ~CUDABatchManager();

    // 添加一个PT作为任务到当前批次
    void AddTaskFromPT(const PT& pt);

    // 当批次达到阈值时，处理当前批次
    void FlushIfReady();

    // 强制处理所有剩余的任务
    void FlushAll();

private:
    PriorityQueue& p_queue; // 对主优先队列的引用
    std::vector<GuessTask> host_tasks; // 主机端的任务队列
    long long total_guesses_in_batch;  // 当前批次中累计的猜测总数
    long long batch_threshold;         // 触发处理的阈值

    // 准备并运行批处理
    void PrepareAndRunBatch();

    // 释放为批处理分配的GPU内存
    void FreeBatchGPUMemory(/* ... */);
};

// 扩展的PriorityQueue类，现在包含批处理管理器
class PriorityQueue_CUDA : public PriorityQueue {
public:
    PriorityQueue_CUDA();
    ~PriorityQueue_CUDA();
    
    // PopNext_CUDA现在将任务添加到批处理管理器
    void PopNext_CUDA();

    // Flush_CUDA用于在循环结束时处理所有剩余的任务
    void Flush_CUDA();

private:
    CUDABatchManager* batch_manager;
};

#endif // GUESSING_CUDA_H

