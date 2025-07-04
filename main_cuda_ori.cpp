#include "PCFG.h"
#include <chrono>
#include "md5.h"
#include <iomanip>
#include "config.h"
#include "guessing_cuda.h"
#include <string_view>


// avx
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

void perform_hash_calculation(PriorityQueue& q, double& time_hash) {
    auto start_hash = system_clock::now();
    
    #ifdef USING_SIMD
    // 使用AVX进行批处理
    string_view passwords[8];
    for(size_t i = 0; i < q.guesses.size(); i += 8) {
        // 改成 string_view
        for (int j = 0; j < 8; ++j) {
            if (i + j < q.guesses.size()) {
                passwords[j] = q.guesses[i + j];
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
        MD5Hash(pw, state);
    }
    #endif
    
    // 计算哈希时间
    auto end_hash = system_clock::now();
    auto duration = duration_cast<microseconds>(end_hash - start_hash);
    time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
}

#endif

//BUG 需要确保 产生的 猜测 每次 都 大于 1000000 ， 因为我只管理了一个 这个指针
// 然后需要每次生成猜测， 都把所有hash掉。
// #ifndef USING_POOL
//BUGFIXUP 不需要了, 原来 如果并发 launch_gpu_kernel 的时候，只有一个全局指针用。。会出事。
// extern char* h_guess_buffer;
// #endif


#ifdef TIME_COUNT
extern double time_add_task;
extern double time_launch_task;
extern double time_before_launch;
extern double time_after_launch;
extern double time_all_batch;
extern double time_string_process;
extern double time_memcpy_toh;
extern double time_gpu_kernel;
extern double time_popnext_non_generate;  // 新增
extern double time_calprob;  // 新增
#endif

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
#ifdef _WIN32
    system("chcp 65001 > nul");
#endif

#ifdef USING_POOL
    // 在main函数开始时初始化线程池
    thread_pool = make_unique<ThreadPool>(THREAD_NUM);
#endif

    task_manager = new TaskManager();

    // 添加时间戳
    auto now = system_clock::now();
    auto now_time = system_clock::to_time_t(now);

#ifdef _WIN32
    #ifdef USING_SIMD
    cout << "\n--- WIN SIMD CUDA MD5 实验批次 [" << std::ctime(&now_time) << "] ---\n";
    #else
    cout << "\n--- WIN 标准CUDA MD5 实验批次 [" << std::ctime(&now_time) << "] ---\n";
    #endif
#else
    #ifdef USING_SIMD
    cout << "\n--- SIMD MD5 CUDA 实验批次 [" << std::ctime(&now_time) << "] ---\n";
    #else
    cout << "\n--- 标准 MD5 CUDA 实验批次 [" << std::ctime(&now_time) << "] ---\n";
    #endif
#endif
    
    // 训练模型（只需一次）
    PriorityQueue q;
    auto start_train = system_clock::now();
// 将windows下用的main.cpp合并进来了
#ifdef _WIN32

#ifdef USING_SMALL
    q.m.train(".\\guessdata\\small_Rockyou-singleLined-full.txt");
#else
    q.m.train(".\\guessdata\\Rockyou-singleLined-full.txt");
#endif
#else
    q.m.train("./guessdata/Rockyou-singleLined-full.txt");
#endif



    q.m.order();
    // 传输数据到gpu， 但是算在训练时间里？ 每次又不需要重置...
#ifdef TIME_COUNT
auto start_transfer = system_clock::now();
#endif
    init_gpu_ordered_values_data(gpu_data,q);
#ifdef TIME_COUNT
auto end_transfer = system_clock::now();
auto duration_transfergpu = duration_cast<microseconds>(end_transfer - start_transfer);
double time_transfergpu = double(duration_transfergpu.count()) * microseconds::period::num / microseconds::period::den;
cout << "time transfer gpu :" << time_transfergpu << endl;
#endif

    // 初始化映射表（只初始化一次，不管是否使用线程池）
    SegmentLengthMaps::getInstance()->init(q);
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
        cout << "猜测上限: " << GENERATE_N << ", 批处理大小: " << NUM_PER_HASH << "， GPU批处理大小：" << GPU_BATCH_SIZE << ", 每个线程处理的guess数："<< GUESS_PER_THREAD;
#ifdef USING_POOL
        cout << "， 线程池线程数: "<< THREAD_NUM;
#endif
        cout << endl;
        cout << "==========================================" << endl;
        

#ifdef DEBUG
cout <<" 开始初始化队列" <<endl;

#endif

#ifdef TIME_COUNT
auto init_time_start = system_clock::now();
#endif
        // 重置队列
        q.init();
        q.guesses.clear();

#ifdef TIME_COUNT
auto init_time_end = system_clock::now();
auto duration_train = duration_cast<microseconds>(init_time_end - init_time_start);
double init_time = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

#endif


#ifdef DEBUG
cout <<" 初始化队列完毕" <<endl;

#endif

        double time_hash = 0;  // 用于MD5哈希的时间
        double time_guess = 0; // 哈希和猜测的总时长
#ifdef TIME_COUNT
        double time_pop_next = 0;
#endif
        auto start = system_clock::now();
        int history = 0;

        while (!q.priority.empty()) {
#ifdef TIME_COUNT
auto start_pop_next = system_clock::now();
#endif

#ifdef DEBUG
cout <<"1"<<endl;
#endif
            q.PopNext();


#ifdef DEBUG
cout <<" 2" <<endl;

#endif
#ifdef TIME_COUNT
auto end_pop_next = system_clock::now();
auto duration_pop_next = duration_cast<microseconds>(end_pop_next - start_pop_next);
time_pop_next += double(duration_pop_next.count()) * microseconds::period::num / microseconds::period::den;
#endif
            int check_guess_count;
            {
                std::lock_guard<std::mutex> lock(main_data_mutex);
                check_guess_count = q.guesses.size();
            }

            // 检查是否达到当前实验的猜测上限
            if (history + check_guess_count > GENERATE_N) {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;

                    cout << "\n--- 实验结果 ---" << endl;
                    cout << "猜测上限: " << GENERATE_N << ", 批处理大小: " << NUM_PER_HASH << endl;
                    cout << "Guesses generated: " << history + check_guess_count << endl;
                    cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
                    cout << "Hash time: " << time_hash << " seconds" << endl;
                    cout << "Train time: " << time_train << " seconds" << endl;
                    cout << "Total time: " << time_guess << " seconds" << endl;

#ifdef TIME_COUNT
cout << "time all pop_next: " << time_pop_next << " seconds" << endl;
cout << "time popnext_non_generate: " << time_popnext_non_generate << " seconds" << endl;
cout << "time calprob: " << time_calprob << " seconds" << endl;  // 新增
cout << "time gpu_kernel: " << time_gpu_kernel << " seconds" << endl;
cout << "time_add_task: " << time_add_task << " seconds" << endl;
cout << "time_launch_task: " << time_launch_task << " seconds" << endl;
cout << "time_before_launch: " << time_before_launch << " seconds" << endl;
cout << "time_after_launch: " << time_after_launch << " seconds" << endl;
cout << "time_string_process: " << time_string_process << " seconds" << endl;
cout << "time_memcpy_toh: " << time_memcpy_toh << " seconds" << endl;
cout << "init_time: " << init_time << " seconds" << endl <<endl;
cout << "time_all_batch: " << time_all_batch << " seconds" << endl <<endl;

#endif
                    cout << "-------------------" << endl;
                break;
            }

            // 检查是否需要进行哈希计算
#ifdef USING_POOL
            int current_guess_count;
            {
                std::lock_guard<std::mutex> lock(main_data_mutex);
                current_guess_count = q.guesses.size();
            }
            
            if (current_guess_count >= NUM_PER_HASH) {
                // 等待所有异步任务完成（需要实现等待机制）
                
                // 执行哈希计算和缓冲区清理
                {
                    std::lock_guard<std::mutex> lock1(main_data_mutex);
                    std::lock_guard<std::mutex> lock2(gpu_buffer_mutex);
                    
                    perform_hash_calculation(q, time_hash);
                    
                    // 释放所有GPU缓冲区
                    for (char* buffer : pending_gpu_buffers) {
                        delete[] buffer;
                    }
                    pending_gpu_buffers.clear();
                    
                    // 更新历史记录并清空guesses
                    history += q.guesses.size();
                    q.guesses.clear();
                }
            }
#else
            if (q.guesses.size() >= NUM_PER_HASH) {
                perform_hash_calculation(q, time_hash);
                
                // 记录已经生成的口令总数
                history += q.guesses.size();
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

        // 最后的哈希计算（处理剩余的guesses）
#ifdef USING_POOL
        {
            std::lock_guard<std::mutex> lock1(main_data_mutex);
            std::lock_guard<std::mutex> lock2(gpu_buffer_mutex);
            
            if (!q.guesses.empty()) {
                perform_hash_calculation(q, time_hash);
                history += q.guesses.size();
            }
            
            // 清理剩余GPU缓冲区
            for (char* buffer : pending_gpu_buffers) {
                delete[] buffer;
            }
            pending_gpu_buffers.clear();
        }
#else
        if (!q.guesses.empty()) {
            perform_hash_calculation(q, time_hash);
            history += q.guesses.size();
        }
        
        // 清理剩余GPU缓冲区（非线程池模式）
        {
            std::lock_guard<std::mutex> lock(gpu_buffer_mutex);
            for (char* buffer : pending_gpu_buffers) {
                delete[] buffer;
            }
            pending_gpu_buffers.clear();
        }
#endif

        // 清理TaskManager
        task_manager->clean();
        
#ifdef USING_POOL
        // 每轮实验结束后，等待线程池所有任务完成并清理
        cout << "等待线程池任务完成..." << endl;
        try {
            // 按照 main_pool.cpp 的方式清理并重建线程池
            thread_pool.reset();  // 销毁当前线程池，等待所有任务完成
            thread_pool = make_unique<ThreadPool>(THREAD_NUM);  // 重新创建
            
            // 确保在安全状态下清空相关向量
            {
                std::lock_guard<std::mutex> lock1(main_data_mutex);
                std::lock_guard<std::mutex> lock2(gpu_buffer_mutex);
                q.guesses.clear();
                
                // 清理任何剩余的GPU缓冲区
                for (char* buffer : pending_gpu_buffers) {
                    delete[] buffer;
                }
                pending_gpu_buffers.clear();
            }
        } catch (const std::exception& e) {
            cerr << "重建线程池失败: " << e.what() << endl;
            abort();
        }
        pending_task_count = 0;
#ifdef TIME_COUNT
        time_pop_next = 0;
        time_popnext_non_generate = 0;  // 新增
        time_calprob = 0;  // 新增
        time_gpu_kernel = 0;
        time_add_task = 0;
        time_launch_task = 0;
        time_before_launch = 0;
        time_after_launch = 0;
        time_string_process = 0;
        time_memcpy_toh = 0;
        time_all_batch = 0;
#endif
        cout << "实验 #" << (exp_idx + 1) << " 完成，线程池已清理" << endl;
#endif
    }

    clean_gpu_ordered_values_data(gpu_data);
    
    cout << "\n--- 实验批次结束 ---\n" << endl;
    return 0;
}