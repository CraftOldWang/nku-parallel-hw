#include "PCFG.h"
#include "guessing_cuda.h"
#include <chrono>
#include "md5.h"
#include <iomanip>
#include "config.h"
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;

// 实验配置数组
struct ExperimentConfig {
    int generate_n;     // 猜测上限
    const char* label;  // 实验标签
};

// 在这里定义所有要运行的实验
const ExperimentConfig EXPERIMENTS[] = {
    {10000000, "10M Guesses"},
    {20000000, "20M Guesses"},
    {30000000, "30M Guesses"},
    {40000000, "40M Guesses"},
    {50000000, "50M Guesses"},
    {60000000, "60M Guesses"},
    {70000000, "70M Guesses"},
    {80000000, "80M Guesses"},
    {90000000, "90M Guesses"},
    {100000000, "100M Guesses"},
};

// 要运行的实验数量
const int NUM_EXPERIMENTS = sizeof(EXPERIMENTS) / sizeof(EXPERIMENTS[0]);

// 编译指令如下
// nvcc main_cuda.cpp guessing_cuda.cu train.cpp guessing.cpp md5.cpp -o main_cuda
// 或者使用CMake: cmake -B build -S . -f CMakeLists_cuda.txt && cmake --build build

int main()
{
#ifdef _WIN32
    system("chcp 65001 > nul");
#endif

    // 检查CUDA设备
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        cout << "错误: 未找到CUDA设备!" << endl;
        return -1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "使用CUDA设备: " << prop.name << endl;
    cout << "计算能力: " << prop.major << "." << prop.minor << endl;

    // 添加时间戳
    auto now = system_clock::now();
    time_t now_time = system_clock::to_time_t(now);
    cout << "\n--- CUDA MD5 实验批次 [" << std::ctime(&now_time) << "] ---\n";
    
    // 训练模型（只需一次）
    PCFG pcfg;
    auto start_train = system_clock::now();
    
#ifdef _WIN32
    pcfg.train(".\\guessdata\\Rockyou-singleLined-full.txt");
#else
    pcfg.train("/guessdata/Rockyou-singleLined-full.txt");
#endif
    pcfg.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    double time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
    
    cout << "模型训练完成，耗时: " << time_train << " 秒" << endl;
    
    // 为每个实验配置运行测试
    for (int exp_idx = 0; exp_idx < NUM_EXPERIMENTS; exp_idx++) {
        // 获取当前实验配置
        int GENERATE_N = EXPERIMENTS[exp_idx].generate_n;
        const char* LABEL = EXPERIMENTS[exp_idx].label;
        
        cout << "\n==========================================" << endl;
        cout << "实验 #" << (exp_idx + 1) << ": " << LABEL << endl;
        cout << "目标猜测数: " << GENERATE_N << ", 批处理阈值: " << CUDA_BATCH_THRESHOLD << endl;
        cout << "==========================================" << endl;
        
        // 初始化CUDA优先队列
        PriorityQueue_CUDA q(pcfg);

        auto start_guess = system_clock::now();
        
        // 生成猜测，直到达到或超过目标数量
        while(q.get_total_guesses() < GENERATE_N) {
            q.PopNext_CUDA();
            if (q.is_empty()) { // 如果队列为空，说明无法生成更多猜测
                cout << "警告: 优先队列已空，无法生成更多猜测。" << endl;
                break;
            }
        }

        // 处理批处理队列中剩余的所有任务
        q.Flush_CUDA();

        auto end_guess = system_clock::now();
        auto duration_guess = duration_cast<microseconds>(end_guess - start_guess);
        double time_guess = double(duration_guess.count()) * microseconds::period::num / microseconds::period::den;

        long long final_guess_count = q.get_total_guesses();

        cout << "\n--- 结果 ---" << endl;
        cout << "总生成猜测数: " << final_guess_count << endl;
        cout << "总生成耗时: " << fixed << setprecision(4) << time_guess << " 秒" << endl;
        if (time_guess > 0) {
            cout << "生成速度: " << fixed << setprecision(2) << (final_guess_count / time_guess) / 1000000.0 << " M/s" << endl;
        }

        // 可选：在这里添加MD5哈希和验证逻辑
        // 注意：q.guesses 现在包含了所有生成的密码，可能会非常大
        // 如果需要，可以分块处理以避免内存问题
        cout << "(MD5哈希验证部分已在此版本中省略)" << endl;

        cout << "实验 #" << (exp_idx + 1) << " 完成." << endl;
    }

    cout << "\n所有实验完成。" << endl;

    return 0;
}