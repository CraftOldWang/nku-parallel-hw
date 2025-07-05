#include "PCFG.h"
#include <chrono>
#include "md5.h"
#include "md5_avx.h"  // AVX实现的MD5
#include <iomanip>
#include <immintrin.h> // AVX 指令集头文件

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

// 编译指令
// g++ main_avx.cpp train.cpp guessing.cpp md5.cpp md5_avx.cpp -o main_avx -mavx2 -O3

int main()
{
    // 添加时间戳
    auto now = system_clock::now();
    auto now_time = system_clock::to_time_t(now);
    cout << "\n--- AVX MD5 实验批次 [" << std::ctime(&now_time) << "] ---\n";
    
    // 训练模型（只需一次）
    PriorityQueue q;
    auto start_train = system_clock::now();
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
                // cout << "Start performing the hash calculation..." << endl;
                auto start_hash = system_clock::now();
                
                // 使用AVX进行批处理
                for(size_t i = 0; i < q.guesses.size(); i += 8) {
                    string passwords[8] = {"", "", "", "", "", "", "", ""};
                    
                    // 填充密码数组（考虑边界情况）
                    for (int j = 0; j < 8 && (i + j) < q.guesses.size(); ++j) {
                        passwords[j] = q.guesses[i + j];
                    }
                    
                    // 使用AVX版本计算MD5
                    __m256i state[4]; // 8个密码的状态向量
                    MD5Hash_AVX(passwords, state);
                }
                
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