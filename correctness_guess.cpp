#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include "guessing_cuda.h"
#include <string_view>
#include <mutex>
#include "config.h"

// AVX支持
#ifdef USING_SIMD
#include "md5_avx.h"  // AVX实现的MD5
#include <immintrin.h> // AVX 指令集头文件
#endif

using namespace std;
using namespace chrono;

// 全局变量（无论是否使用线程池都需要）
mutex main_data_mutex;           // 保护主要数据结构
mutex gpu_buffer_mutex;         // 保护GPU缓冲区管理
vector<char*> pending_gpu_buffers;  // 等待释放的GPU缓冲区指针

#ifdef USING_POOL
#include "ThreadPool.h"
#include <atomic>
std::atomic<int> pending_task_count(0);  // 初始化为0
std::unique_ptr<ThreadPool> thread_pool;  // 声明全局变量，但不初始化

void perform_hash_calculation_with_test(PriorityQueue& q, double& time_hash, 
                                       const unordered_set<string>& test_set, 
                                       int& cracked) {
    auto start_hash = system_clock::now();
    
    #ifdef USING_SIMD
    // 使用AVX进行批处理
    string_view passwords[8];
    for(size_t i = 0; i < q.guesses.size(); i += 8) {
        for (int j = 0; j < 8; ++j) {
            if (i + j < q.guesses.size()) {
                passwords[j] = q.guesses[i + j];
                // 检测正确性
                if (test_set.find(string(passwords[j])) != test_set.end()) {
                    cracked++;
                }
            } else {
                passwords[j] = "";  // 用空字符串占位，避免越界
            }
        }
        
        // 使用AVX版本计算MD5
        alignas(32) __m256i state[4]; // 8个密码的状态向量
        MD5Hash_AVX(passwords, state);
    }
    #else
    // 使用标准MD5计算
    bit32 state[4];
    for (string_view pw : q.guesses) {
        // 检测正确性
        if (test_set.find(string(pw)) != test_set.end()) {
            cracked++;
        }
        MD5Hash(pw, state);
    }
    #endif
    
    // 计算哈希时间
    auto end_hash = system_clock::now();
    auto duration = duration_cast<microseconds>(end_hash - start_hash);
    time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
}

#else
// 非线程池模式的哈希计算函数
void perform_hash_calculation_with_test(PriorityQueue& q, double& time_hash, 
                                       const unordered_set<string>& test_set, 
                                       int& cracked) {
    auto start_hash = system_clock::now();
    
    #ifdef USING_SIMD
    // 使用AVX进行批处理
    string_view passwords[8];
    for(size_t i = 0; i < q.guesses.size(); i += 8) {
        for (int j = 0; j < 8; ++j) {
            if (i + j < q.guesses.size()) {
                passwords[j] = q.guesses[i + j];
                // 检测正确性
                if (test_set.find(string(passwords[j])) != test_set.end()) {
                    cracked++;
                }
            } else {
                passwords[j] = "";  // 用空字符串占位，避免越界
            }
        }
        
        // 使用AVX版本计算MD5
        alignas(32) __m256i state[4]; // 8个密码的状态向量
        MD5Hash_AVX(passwords, state);
    }
    #else
    // 使用标准MD5计算
    bit32 state[4];
    for (string_view pw : q.guesses) {
        // 检测正确性
        if (test_set.find(string(pw)) != test_set.end()) {
            cracked++;
        }
        MD5Hash(pw, state);
    }
    #endif
    
    // 计算哈希时间
    auto end_hash = system_clock::now();
    auto duration = duration_cast<microseconds>(end_hash - start_hash);
    time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
}
#endif

// #define USING_SMALL

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2

int main()
{
#ifdef DEBUG
    cout << "=== DEBUG: 程序开始 ===" << endl;
#endif

#ifdef _WIN32
    system("chcp 65001 > nul");
#endif

#ifdef DEBUG
    cout << "DEBUG: 系统初始化完成" << endl;
#endif

#ifdef USING_POOL
    // 在main函数开始时初始化线程池
#ifdef DEBUG
    cout << "DEBUG: 开始初始化线程池，线程数: " << THREAD_NUM << endl;
#endif
    thread_pool = make_unique<ThreadPool>(THREAD_NUM);
#ifdef DEBUG
    cout << "DEBUG: 线程池初始化完成" << endl;
#endif
#endif

#ifdef DEBUG
    cout << "DEBUG: 开始创建TaskManager" << endl;
#endif
    task_manager = new TaskManager();
#ifdef DEBUG
    cout << "DEBUG: TaskManager创建完成" << endl;
#endif

    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    int cracked = 0;       // 破解的密码数量
    
    PriorityQueue q;
#ifdef DEBUG
    cout << "DEBUG: 开始模型训练" << endl;
#endif
    auto start_train = system_clock::now();
#ifdef _WIN32
    #ifdef USING_SMALL
#ifdef DEBUG
    cout << "DEBUG: 使用小数据集训练模型" << endl;
#endif
    q.m.train(".\\guessdata\\small_Rockyou-singleLined-full.txt");
    #else
#ifdef DEBUG
    cout << "DEBUG: 使用完整数据集训练模型" << endl;
#endif
    q.m.train(".\\guessdata\\Rockyou-singleLined-full.txt");
    #endif
#else
#ifdef DEBUG
    cout << "DEBUG: Linux环境，开始训练模型" << endl;
#endif
    q.m.train("./guessdata/Rockyou-singleLined-full.txt");
#endif

#ifdef DEBUG
    cout << "DEBUG: 模型训练完成，开始排序" << endl;
#endif
    q.m.order();

#ifdef DEBUG
    cout << "DEBUG: 开始初始化GPU数据" << endl;
#endif
    init_gpu_ordered_values_data(gpu_data,q);
#ifdef DEBUG
    cout << "DEBUG: GPU数据初始化完成" << endl;
#endif
    
    // 初始化映射表（统一的方式）
#ifdef DEBUG
    cout << "DEBUG: 开始初始化映射表" << endl;
#endif
    SegmentLengthMaps::getInstance()->init(q);
    PTMaps::getInstance()->init(q);

#ifdef DEBUG
    cout << "DEBUG: 映射表初始化完成" << endl;
#endif
    
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

#ifdef DEBUG
    cout << "DEBUG: 训练阶段完成，耗时: " << time_train << " 秒" << endl;
#endif

    // 加载一些测试数据
    // 加载测试数据时需要注意生命周期
#ifdef DEBUG
    cout << "DEBUG: 开始加载测试数据" << endl;
#endif
    vector<string> test_passwords;  // 用于保存实际的字符串数据
    unordered_set<string> test_set;

#ifdef _WIN32
    #ifdef USING_SMALL
    ifstream test_data(".\\guessdata\\small_Rockyou-singleLined-full.txt");
    #else
    ifstream test_data(".\\guessdata\\Rockyou-singleLined-full.txt");
    #endif
#else
    ifstream test_data("./guessdata/Rockyou-singleLined-full.txt");
#endif
    int test_count=0;
    string pw;
    while(test_data>>pw)
    {   
        test_count+=1;
        test_passwords.push_back(pw);  // 保存实际数据
        test_set.insert(test_passwords.back());  // 插入 string_view
        if (test_count>=1000000)
        {
            break;
        }
    }

#ifdef DEBUG
    cout << "DEBUG: 测试数据加载完成，共加载 " << test_count << " 个密码" << endl;
#endif

#ifdef DEBUG
    cout << "DEBUG: 开始初始化优先队列" << endl;
#endif
    q.init();
#ifdef DEBUG
    cout << "DEBUG: 优先队列初始化完成，队列大小: " << q.priority.size() << endl;
#endif
    cout << "Testing password cracking with " << test_count << " passwords..." << endl;
    
#ifdef USING_POOL
    cout << "Using thread pool with " << THREAD_NUM << " threads" << endl;
#ifdef USING_SIMD
    cout << "SIMD optimization enabled" << endl;
#endif
#else
    cout << "Using synchronous mode" << endl;
#ifdef USING_SIMD
    cout << "SIMD optimization enabled" << endl;
#endif
#endif

    int curr_num = 0;
    auto start = system_clock::now();
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;
    
#ifdef DEBUG
    cout << "DEBUG: 开始主循环，优先队列初始大小: " << q.priority.size() << endl;
#endif
    
    // std::ofstream a("./files/results.txt");
    while (!q.priority.empty())
    {
#ifdef DEBUG
        static int loop_count = 0;
        loop_count++;
        if (loop_count % 1000 == 0) {
            cout << "DEBUG: 主循环第 " << loop_count << " 次迭代，队列大小: " << q.priority.size() << endl;
        }
#endif
        
#ifdef DEBUG
        if (loop_count <= 5) {
            cout << "DEBUG: 准备调用 PopNext(), 第 " << loop_count << " 次" << endl;
        }
#endif
        q.PopNext();
#ifdef DEBUG
        if (loop_count <= 5) {
            cout << "DEBUG: PopNext() 完成，当前 guesses 大小: " << q.guesses.size() << endl;
        }
#endif
        
        int check_guess_count;
        {
            std::lock_guard<std::mutex> lock(main_data_mutex);
            check_guess_count = q.guesses.size();
        }
        
#ifdef DEBUG
        if (loop_count <= 5) {
            cout << "DEBUG: 获取到 check_guess_count: " << check_guess_count << endl;
        }
#endif
        
        if (check_guess_count - curr_num >= 100000)
        {
#ifdef DEBUG
            cout << "DEBUG: 达到报告阈值，当前猜测数: " << history + check_guess_count << endl;
#endif
            cout << "Guesses generated: " << history + check_guess_count << endl;
            curr_num = check_guess_count;

            // 在此处更改实验生成的猜测上限
            int generate_n = 10000000;
            if (history + check_guess_count > generate_n)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
                cout << "Hash time: " << time_hash << " seconds" << endl;
                cout << "Train time: " << time_train << " seconds" << endl;
                cout << "Cracked: " << cracked << " passwords" << endl;
                cout << "Success rate: " << (double)cracked / test_count * 100 << "%" << endl;
                break;
            }
        }
        
        // 检查是否需要进行哈希计算
#ifdef USING_POOL
        int current_guess_count;
        {
            std::lock_guard<std::mutex> lock(main_data_mutex);
            current_guess_count = q.guesses.size();
        }
        
        if (current_guess_count > 1000000)
        {
#ifdef DEBUG
            cout << "DEBUG: 线程池模式 - 达到哈希阈值，准备进行哈希计算" << endl;
#endif
            // 执行哈希计算和缓冲区清理
            {
                std::lock_guard<std::mutex> lock1(main_data_mutex);
                std::lock_guard<std::mutex> lock2(gpu_buffer_mutex);
                
#ifdef DEBUG
                cout << "DEBUG: 获得锁，开始哈希计算，猜测数: " << q.guesses.size() << endl;
#endif
                perform_hash_calculation_with_test(q, time_hash, test_set, cracked);
#ifdef DEBUG
                cout << "DEBUG: 哈希计算完成，破解数: " << cracked << endl;
#endif
                
                // 释放所有GPU缓冲区
                for (char* buffer : pending_gpu_buffers) {
                    delete[] buffer;
                }
                pending_gpu_buffers.clear();
                
                // 更新历史记录并清空guesses
                history += q.guesses.size();
                curr_num = 0;
                q.guesses.clear();
            }
        }
#else
        if (check_guess_count > 1000000)
        {
#ifdef DEBUG
            cout << "DEBUG: 非线程池模式 - 达到哈希阈值，准备进行哈希计算" << endl;
#endif
            perform_hash_calculation_with_test(q, time_hash, test_set, cracked);
#ifdef DEBUG
            cout << "DEBUG: 非线程池模式哈希计算完成，破解数: " << cracked << endl;
#endif
            
            // 记录已经生成的口令总数
            history += check_guess_count;
            curr_num = 0;
            q.guesses.clear();
            
            // 统一使用缓冲区管理，释放所有GPU缓冲区
            {
                std::lock_guard<std::mutex> lock(gpu_buffer_mutex);
                for (char* buffer : pending_gpu_buffers) {
                    delete[] buffer;
                }
                pending_gpu_buffers.clear();
            }
        }
#endif
    }
    
#ifdef DEBUG
    cout << "DEBUG: 主循环结束，开始最终清理" << endl;
#endif
    
    // 最后的哈希计算（处理剩余的guesses）
#ifdef USING_POOL
    {
#ifdef DEBUG
        cout << "DEBUG: 线程池模式最终清理开始" << endl;
#endif
        std::lock_guard<std::mutex> lock1(main_data_mutex);
        std::lock_guard<std::mutex> lock2(gpu_buffer_mutex);
        
        if (!q.guesses.empty()) {
#ifdef DEBUG
            cout << "DEBUG: 处理剩余的 " << q.guesses.size() << " 个猜测" << endl;
#endif
            perform_hash_calculation_with_test(q, time_hash, test_set, cracked);
            history += q.guesses.size();
        }
        
        // 清理剩余GPU缓冲区
#ifdef DEBUG
        cout << "DEBUG: 清理 " << pending_gpu_buffers.size() << " 个GPU缓冲区" << endl;
#endif
        for (char* buffer : pending_gpu_buffers) {
            delete[] buffer;
        }
        pending_gpu_buffers.clear();
    }
    
    // 等待线程池任务完成
    cout << "等待线程池任务完成..." << endl;
#ifdef DEBUG
    cout << "DEBUG: 销毁线程池" << endl;
#endif
    thread_pool.reset();  // 销毁线程池，等待所有任务完成
#ifdef DEBUG
    cout << "DEBUG: 线程池销毁完成" << endl;
#endif
#else
    if (!q.guesses.empty()) {
#ifdef DEBUG
        cout << "DEBUG: 非线程池模式最终清理，处理剩余的 " << q.guesses.size() << " 个猜测" << endl;
#endif
        perform_hash_calculation_with_test(q, time_hash, test_set, cracked);
        history += q.guesses.size();
    }
    
    // 清理剩余GPU缓冲区（非线程池模式）
    {
#ifdef DEBUG
        cout << "DEBUG: 非线程池模式清理 " << pending_gpu_buffers.size() << " 个GPU缓冲区" << endl;
#endif
        std::lock_guard<std::mutex> lock(gpu_buffer_mutex);
        for (char* buffer : pending_gpu_buffers) {
            delete[] buffer;
        }
        pending_gpu_buffers.clear();
    }
#endif

#ifdef DEBUG
    cout << "DEBUG: 开始最终资源清理" << endl;
#endif
    
    // 输出最终结果
    cout << "\n=== 最终测试结果 ===" << endl;
    cout << "总猜测数量: " << history << endl;
    cout << "测试密码数量: " << test_count << endl;
    cout << "成功破解: " << cracked << " 个密码" << endl;
    cout << "破解成功率: " << (double)cracked / test_count * 100 << "%" << endl;
    cout << "猜测时间: " << time_guess - time_hash << " 秒" << endl;
    cout << "哈希时间: " << time_hash << " 秒" << endl;
    cout << "训练时间: " << time_train << " 秒" << endl;
    cout << "总时间: " << time_guess << " 秒" << endl;
    
    task_manager->clean();
    clean_gpu_ordered_values_data(gpu_data);
    
    return 0;
}
