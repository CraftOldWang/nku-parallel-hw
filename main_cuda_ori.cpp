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

#include "ThreadPool.h"
#include <atomic>
std::atomic<int> pending_task_count(0);  // 初始化为0


int expected_guess_num  = 0 ;

std::unique_ptr<ThreadPool> thread_pool;  // 声明全局变量，但不初始化
std::unique_ptr<ThreadPool> hash_thread_pool; // Hash专用线程池

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
void perform_hash_calculation_parallel(PriorityQueue& q, double& time_hash) {
    auto start_hash = system_clock::now();
    
    const size_t total_guesses = q.guesses.size();
    
    
    
    if (total_guesses < 1000) {
        // 串行SIMD处理
        for (size_t i = 0; i < total_guesses; i += 8) {
            string_view passwords[8];
            for (int j = 0; j < 8; ++j) {
                passwords[j] = (i + j < total_guesses) ? q.guesses[i + j] : "";
            }
            alignas(32) __m256i state[4];
            MD5Hash_AVX(passwords, state);
        }
    } else {

        const size_t num_threads = std::min(static_cast<size_t>(HASH_THREAD_NUM), total_guesses / 100);
        const size_t chunk_size = total_guesses / num_threads;
        
        std::vector<std::future<void>> hash_futures;
        
        for (size_t t = 0; t < num_threads; ++t) {
            size_t start_idx = t * chunk_size;
            size_t end_idx = (t == num_threads - 1) ? total_guesses : (t + 1) * chunk_size;
            
            hash_futures.push_back(hash_thread_pool->enqueue([&q, start_idx, end_idx]() {
                // 每个线程处理自己的chunk，使用AVX
                for (size_t i = start_idx; i < end_idx; i += 8) {
                    string_view passwords[8];
                    for (int j = 0; j < 8; ++j) {
                        if (i + j < end_idx) {
                            passwords[j] = q.guesses[i + j];
                        } else {
                            passwords[j] = "";
                        }
                    }
                    alignas(32) __m256i state[4];
                    MD5Hash_AVX(passwords, state);
                }
            }));
        }
        
        // 等待所有hash任务完成
        for (auto& future : hash_futures) {
            future.wait();
        }
    }
    
    auto end_hash = system_clock::now();
    auto duration = duration_cast<microseconds>(end_hash - start_hash);
    time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
}

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

    // 亿级别 (感觉跑hash 需要比较多秒？线程池给多一点吧)
    {100000000, 1000000, "数据集/小批次"},
    {150000000, 1000000, "数据集/小批次"},
    {200000000, 1000000, "数据集/小批次"},
    {250000000, 1000000, "数据集/小批次"},
    {300000000, 1000000, "数据集/小批次"},
    {350000000, 1000000, "数据集/小批次"},
    {400000000, 1000000, "数据集/小批次"},
    {450000000, 1000000, "数据集/小批次"},
    {500000000, 1000000, "数据集/小批次"},

    // 十亿级别
    {1000000000, 1000000, "数据集/小批次"},

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
    hash_thread_pool = make_unique<ThreadPool>(HASH_THREAD_NUM); // Hash计算

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


    q.m.initMapping(q);
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


    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    double time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
    
    cout << "模型训练完成，耗时: " << time_train << " 秒" << endl;
    
    // 为每个实验配置运行测试
    for (int exp_idx = 0; exp_idx < NUM_EXPERIMENTS; exp_idx++) {
        // 获取当前实验配置
        int GENERATE_N = EXPERIMENTS[exp_idx].generate_n;
        // 历史遗留。。。没用了。
        int NUM_PER_HASH = EXPERIMENTS[exp_idx].batch_size;
        const char* LABEL = EXPERIMENTS[exp_idx].label;
        
        cout << "\n==========================================" << endl;
        cout << "实验 #" << (exp_idx + 1) << ": " << LABEL << endl;
        cout << "猜测上限: " << GENERATE_N << ", 批处理大小: " << NUM_PER_HASH << "， GPU批处理大小：" << GPU_BATCH_SIZE << ", 每个线程处理的guess数："<< GUESS_PER_THREAD;
#ifdef USING_POOL
        cout << "， guess生成线程池线程数: "<< THREAD_NUM;
        cout << ", hash 线程池线程数: " << HASH_THREAD_NUM;
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
        expected_guess_num = 0; 
        while (!q.priority.empty()) {
#ifdef TIME_COUNT
auto start_pop_next = system_clock::now();
#endif

            if (expected_guess_num < GENERATE_N) {
                q.PopNext();
            } else {
#ifdef SLEEP_COUT
                cout << "sleep for a while\n";
#endif
                // 为了看能否见缝插针地hash ， 这里不敢wait
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

#ifdef TIME_COUNT
auto end_pop_next = system_clock::now();
auto duration_pop_next = duration_cast<microseconds>(end_pop_next - start_pop_next);
time_pop_next += double(duration_pop_next.count()) * microseconds::period::num / microseconds::period::den;
#endif


            // 检查是否需要进行哈希计算
            bool should_hash;
            {
                std::lock_guard<std::mutex> lock1(main_data_mutex);
                std::lock_guard<std::mutex> lock2(gpu_buffer_mutex);      
                should_hash = !pending_gpu_buffers.empty();
            }
            
            //BUG 要设置每个 GPU 批次大于这个值.... 或者干脆别设置这个
            if (should_hash) {                
                // 执行哈希计算和缓冲区清理
                {
                    std::lock_guard<std::mutex> lock1(main_data_mutex);
                    std::lock_guard<std::mutex> lock2(gpu_buffer_mutex);
                    
                    perform_hash_calculation_parallel(q, time_hash);
                    
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



            // 检查是否达到当前实验的猜测上限
            if (history  >= GENERATE_N) {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;

                    cout << "\n--- 实验结果 ---" << endl;
                    cout << "猜测上限: " << GENERATE_N << ", 批处理大小: " << NUM_PER_HASH << endl;
                    cout << "Guesses generated: " << history  << endl;
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

        }
        // 清理TaskManager
        task_manager->clean();
        
        // 每轮实验结束后，等待线程池所有任务完成并清理
        cout << "等待线程池任务完成..." << endl;
        try {
            // 按照 main_pool.cpp 的方式清理并重建线程池
            thread_pool.reset();  // 销毁当前线程池，等待所有任务完成
            thread_pool = make_unique<ThreadPool>(THREAD_NUM);  // 重新创建

            hash_thread_pool.reset();
            hash_thread_pool = make_unique<ThreadPool>(HASH_THREAD_NUM);

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
        
    }

    clean_gpu_ordered_values_data(gpu_data);
    
    cout << "\n--- 实验批次结束 ---\n" << endl;
    return 0;
}