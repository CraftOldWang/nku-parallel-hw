#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <memory> // For std::unique_ptr
#include <iomanip> // For std::setprecision
#include "config.h" // 包含配置文件
#include "md5.h"
#include "guessing_cuda.h" // 包含猜测相关的CUDA实现
#include "PCFG.h" // 包含PCFG相关的类和函数

// ... (包含你的其他头文件如 PriorityQueue, TaskManager, MD5, GPU anagement等)

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::chrono::system_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// --- 1. 抽象配置和结果 ---
// 使数据结构更清晰
struct ExperimentConfig {
    int generate_n;   // 猜测上限
    int batch_size;   // 一次处理的口令数量
    const char* label;// 实验标签
};

struct ExperimentResult {
    long long guesses_generated = 0;
    double guess_time = 0.0;
    double hash_time = 0.0;

    double total_time() const { return guess_time + hash_time; }
};

// 假设的全局实验配置
const ExperimentConfig EXPERIMENTS[] = {
    {10000000, 1000000, "数据集/小批次"},
    {15000000, 1000000, "数据集/小批次"},
    {20000000, 1000000, "数据集/小批次"},
    {25000000, 1000000, "数据集/小批次"},
    {30000000, 1000000, "数据集/小批次"},
    {35000000, 1000000, "数据集/小批次"},
    {40000000, 1000000, "数据集/小批次"},
    {45000000, 1000000, "数据集/小批次"},
    {50000000, 1000000, "数据集/小批次"},
};
const int NUM_EXPERIMENTS = sizeof(EXPERIMENTS) / sizeof(ExperimentConfig);


// --- 2. 封装实现细节 ---

// 封装平台特定的文件路径
string GetTrainingDataPath() {
#ifdef _WIN32
    return ".\\guessdata\\Rockyou-singleLined-full.txt";
#else
    return "/guessdata/Rockyou-singleLined-full.txt";
#endif
}

// 封装批处理逻辑，隐藏SIMD与否的细节
void ProcessBatch(const vector<string>& guesses) {
#ifdef USING_SIMD
    //TODO 这里没弄过来，不过估计懒得做SIMD了
    // ... 你的SIMD MD5计算逻辑 ...
    // 例如:
    // alignas(16) uint32x4_t state[4];
    // for(size_t i = 0; i < guesses.size(); i += 4) { ... }
#else
    // 使用标准MD5计算
    bit32 state[4];
    for (const string& pw : guesses) {
        MD5Hash(pw, state);
    }
#endif
}

// --- 3. 分解 main 函数 ---

void SetupEnvironment() {
#ifdef _WIN32
    // 设置Windows控制台代码页为UTF-8，以正确显示中文字符
    system("chcp 65001 > nul");
#endif
}

void PrintExperimentBatchHeader() {
    auto now = system_clock::now();
    auto now_time = system_clock::to_time_t(now);
    cout << "\n======================================================\n";
#ifdef _WIN32
    #ifdef USING_SIMD
        cout << "--- WIN SIMD CUDA MD5 实验批次 [" << std::ctime(&now_time) << "] ---";
    #else
        cout << "--- WIN 标准CUDA MD5 实验批次 [" << std::ctime(&now_time) << "] ---";
    #endif
#else
    #ifdef USING_SIMD
        cout << "--- SIMD MD5 CUDA 实验批次 [" << std::ctime(&now_time) << "] ---";
    #else
        cout << "--- 标准 MD5 CUDA 实验批次 [" << std::ctime(&now_time) << "] ---";
    #endif
#endif
    cout << "======================================================\n";
}

double TrainModel(PriorityQueue& q) {
    cout << "\n正在训练模型...\n";
    auto start_train = system_clock::now();
    
    q.m.train(GetTrainingDataPath());
    q.m.order();
    init_gpu_ordered_values_data(gpu_data, q); // 传输数据到GPU
    
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    double time_train = static_cast<double>(duration_train.count()) * microseconds::period::num / microseconds::period::den;
    
    cout << "模型训练完成，耗时: " << std::fixed << std::setprecision(4) << time_train << " 秒" << endl;
    return time_train;
}

ExperimentResult RunSingleExperiment(const ExperimentConfig& config, PriorityQueue& q) {
    cout << "\n==========================================" << endl;
    cout << "开始实验: " << config.label << endl;
    cout << "配置: 猜测上限=" << config.generate_n << ", 批处理大小=" << config.batch_size << endl;
    cout << "==========================================\n";

    ExperimentResult result;
    q.init(); // 重置队列
    q.guesses.clear();
    
    long long total_guesses_in_exp = 0;
    auto start_time = system_clock::now();

    while (!q.priority.empty() && total_guesses_in_exp < config.generate_n) {
        q.PopNext();

        // 当累积的猜测数量达到批处理大小时，进行处理
        if (q.guesses.size() >= config.batch_size) {
            auto end_guess_gen = system_clock::now();
            ProcessBatch(q.guesses);
            auto end_hash = system_clock::now();

            // 累加时间
            result.guess_time += duration_cast<microseconds>(end_guess_gen - start_time).count();
            result.hash_time += duration_cast<microseconds>(end_hash - end_guess_gen).count();

            total_guesses_in_exp += q.guesses.size();
            q.guesses.clear(); // 清空批次，准备下一次
            start_time = system_clock::now(); // 重置计时器
        }
    }
    
    // 处理循环结束后剩余的不足一个批次的猜测
    if (!q.guesses.empty()) {
        auto end_guess_gen = system_clock::now();
        ProcessBatch(q.guesses);
        auto end_hash = system_clock::now();

        result.guess_time += duration_cast<microseconds>(end_guess_gen - start_time).count();
        result.hash_time += duration_cast<microseconds>(end_hash - end_guess_gen).count();
        
        total_guesses_in_exp += q.guesses.size();
        q.guesses.clear();
    }
    
    result.guesses_generated = total_guesses_in_exp;

    // 将微秒转换为秒
    const double to_seconds = static_cast<double>(microseconds::period::num) / microseconds::period::den;
    result.guess_time *= to_seconds;
    result.hash_time *= to_seconds;

    return result;
}

void PrintExperimentResult(const ExperimentConfig& config, const ExperimentResult& result, double train_time) {
    cout << "\n--- 实验结果 (" << config.label << ") ---" << endl;
    cout << "总共生成猜测: " << result.guesses_generated << endl;
    cout << std::fixed << std::setprecision(4);
    cout << "猜测生成耗时: " << result.guess_time << " 秒" << endl;
    cout << "哈希计算耗时: " << result.hash_time << " 秒" << endl;
    cout << "模型训练耗时: " << train_time << " 秒" << endl;
    cout << "本次实验总耗时: " << result.total_time() << " 秒" << endl;
    cout << "--------------------------------\n";
}


// --- 4. 全新、整洁的 main 函数 ---
int main() {
    SetupEnvironment();
    
    task_manager = new TaskManager();    

    PrintExperimentBatchHeader();

    PriorityQueue q;
    double train_time = TrainModel(q);
    
    // 为每个实验配置运行测试
    for (int i = 0; i < NUM_EXPERIMENTS; ++i) {
        const auto& config = EXPERIMENTS[i];
        ExperimentResult result = RunSingleExperiment(config, q);
        PrintExperimentResult(config, result, train_time);
    }

    // 清理全局变量资源
    cleanup_global_cuda_resources();

    cout << "\n--- 所有实验已完成 ---\n" << endl;
    
    return 0; // unique_ptr 会在这里自动释放 task_manager
}