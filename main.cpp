#include "PCFG.h"
#include <chrono>
#include "md5.h"
#include <iomanip>
#include "config.h"
#include <iostream>
#include <vector>
#include <string>
// #include "guessing_cuda.h" // 包含猜测相关的CUDA实现
#include "PCFG.h" // 包含PCFG相关的类和函数

// 如果定义了使用SIMD，则包含SIMD头文件
// #ifdef USING_SIMD
// #include "md5_simd.h"
// #endif

using namespace std;
using namespace chrono;

// 实验配置数组，每一组包含两个参数：总生成数量和批处理大小
struct ExperimentConfig {
    int generate_n;     // 猜测上限
    int batch_size;     // 一次处理的口令数量
    const char* label;  // 实验标签
};

// 在这里定义所有要运行的实验
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

// 要运行的实验数量
const int NUM_EXPERIMENTS = sizeof(EXPERIMENTS) / sizeof(EXPERIMENTS[0]);

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2

int main()
{
    // system("chcp 65001 > nul");

    // 添加时间戳
    auto now = system_clock::now();
    auto now_time = system_clock::to_time_t(now);

#ifdef _WIN32
    #ifdef USING_SIMD
    cout << "\n--- WIN SIMD MD5 实验批次 [" << std::ctime(&now_time) << "] ---\n";
    #else
    cout << "\n--- WIN 标准 MD5 实验批次 [" << std::ctime(&now_time) << "] ---\n";
    #endif
#else
    #ifdef USING_SIMD
    cout << "\n--- SIMD MD5 实验批次 [" << std::ctime(&now_time) << "] ---\n";
    #else
    cout << "\n--- 标准 MD5 实验批次 [" << std::ctime(&now_time) << "] ---\n";
    #endif
#endif
    
    // 训练模型（只需一次）
    PriorityQueue q;
    auto start_train = system_clock::now();
// 将windows下用的main.cpp合并进来了
#ifdef _WIN32
    q.m.train(".\\guessdata\\Rockyou-singleLined-full.txt");
#else
    q.m.train("./guessdata/Rockyou-singleLined-full.txt");
#endif
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    double time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
    
    cout << "模型训练完成，耗时: " << time_train << " 秒" << endl;
    
    // 为每个实验配置运行测试
    for (int exp_idx = 0; exp_idx < NUM_EXPERIMENTS; exp_idx++) {
        // 获取当前实验配置
        int GENERATE_N = EXPERIMENTS[exp_idx].generate_n;
        int NUM_PER_HASH = EXPERIMENTS[exp_idx].batch_size;
        const char* LABEL = EXPERIMENTS[exp_idx].label;
        
        cout << "\n==========================================" << endl;
        cout << "实验 #" << (exp_idx + 1) << ": " << LABEL << endl;
        cout << "猜测上限: " << GENERATE_N << ", 批处理大小: " << NUM_PER_HASH << endl;
        cout << "==========================================" << endl;
        
        // 重置队列
        q.init();
        q.guesses.clear();

        
        double time_hash = 0;  // 用于MD5哈希的时间
        double time_guess = 0; // 哈希和猜测的总时长
        
        int curr_num = 0;
        auto start = system_clock::now();
        int history = 0;
        
        while (!q.priority.empty())
        {
            q.PopNext();
            q.total_guesses = q.guesses.size();
            if (q.total_guesses - curr_num >= 100000)
            {
                // cout << "Guesses generated: " << history + q.total_guesses << endl;
                curr_num = q.total_guesses;
                
                // 检查是否达到当前实验的猜测上限
                if (history + q.total_guesses > GENERATE_N)
                {
                    auto end = system_clock::now();
                    auto duration = duration_cast<microseconds>(end - start);
                    time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                    
                    cout << "\n--- 实验结果 ---" << endl;
                    cout << "猜测上限: " << GENERATE_N << ", 批处理大小: " << NUM_PER_HASH << endl;
                    cout << "Guesses generated: " << history + q.total_guesses << endl;
                    cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
                    cout << "Hash time: " << time_hash << " seconds" << endl;
                    cout << "Train time: " << time_train << " seconds" << endl;
                    cout << "Total time: " << time_guess << " seconds" << endl;
                    cout << "-------------------" << endl;
                    
                    break;
                }
            }
            
            // 达到批处理大小，进行哈希计算
            if (curr_num > NUM_PER_HASH)
            {
                auto start_hash = system_clock::now();
                
                #ifdef USING_SIMD
                // 使用SIMD进行MD5计算
                #ifdef USING_ALIGNED
                alignas(16) uint32x4_t state[4]; // 每个lane 一个 口令的 一部分 state
                #endif
                #ifndef USING_ALIGNED
                uint32x4_t state[4]; // 每个lane 一个 口令的 一部分 state
                #endif

                #ifdef NOT_USING_STRING_ARR
                size_t i = 0;
                for(; i < q.guesses.size(); i += 4){
                    //HACK string 的copy 也很费时间
                    string &pw1 = q.guesses[i];
                    string &pw2 = q.guesses[i+1];
                    string &pw3 = q.guesses[i+2];
                    string &pw4 = q.guesses[i+3];
    
                    MD5Hash_SIMD(pw1, pw2, pw3, pw4, state);
                }
                bit32 state2[4];
                for (; i < q.guesses.size(); ++i) {
                    MD5Hash(q.guesses[i], state2); // 假设你有个单个处理版本
                }
                #endif

                #ifndef NOT_USING_STRING_ARR
                for(size_t i = 0; i < q.guesses.size(); i += 4){
                    string pw[4] = {"", "", "", ""};
                    for (int j = 0; j < 4 && (i + j) < q.guesses.size(); ++j) {
                        pw[j] = q.guesses[i + j];
                    }
                    MD5Hash_SIMD(pw, state);
                }    
                #endif
                
                #else
                // 使用标准MD5计算
                bit32 state[4];
                
                for (string pw : q.guesses)
                {
                    MD5Hash(pw, state);
                }
                #endif
                
                // 计算哈希时间
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
                
                // 记录已经生成的口令总数
                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
            }
        }
    }
    
    cout << "\n--- 实验批次结束 ---\n" << endl;
    return 0;
}