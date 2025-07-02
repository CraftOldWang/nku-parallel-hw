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

// 如果定义了使用SIMD，则包含SIMD头文件
// #ifdef USING_SIMD
// #include "md5_simd.h"
// #endif
using namespace std;
using namespace chrono;

//BUG 需要确保 产生的 猜测 每次 都 大于 1000000 ， 因为我只管理了一个 这个指针
// 然后需要每次生成猜测， 都把所有hash掉。
extern char* h_guess_buffer;

// #ifdef TIME_COUNT
extern double time_add_task;
extern double time_launch_task;
extern double time_before_launch;
extern double time_after_launch;
extern double time_all_batch;
extern double time_string_process;
extern double time_memcpy_toh;
extern double time_gpu_kernel;
// #endif

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
    q.m.train(".\\guessdata\\Rockyou-singleLined-full.txt");
#else
    q.m.train("./guessdata/Rockyou-singleLined-full.txt");
#endif
    q.m.order();
    // 传输数据到gpu， 但是算在训练时间里？ 每次又不需要重置...
#ifdef TIME_COUNT
auto start_transfer = system_clock::now();
#endif
    init_gpu_ordered_values_data(gpu_data,q);
    task_manager->init_length_maps(q);
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
        int NUM_PER_HASH = EXPERIMENTS[exp_idx].batch_size;
        const char* LABEL = EXPERIMENTS[exp_idx].label;
        
        cout << "\n==========================================" << endl;
        cout << "实验 #" << (exp_idx + 1) << ": " << LABEL << endl;
        cout << "猜测上限: " << GENERATE_N << ", 批处理大小: " << NUM_PER_HASH <<"， GPU批处理大小：" << GPU_BATCH_SIZE  << endl;
        cout << "==========================================" << endl;
        
        // 重置队列
        q.init();
        q.guesses.clear();

        
        double time_hash = 0;  // 用于MD5哈希的时间
        double time_guess = 0; // 哈希和猜测的总时长
#ifdef TIME_COUNT
        double time_pop_next = 0;
        double time_check = 0; // 中间那些检查的时间
#endif
        int curr_num = 0;
        auto start = system_clock::now();
        int history = 0;

double time_guess_part = 0; // 新增，专门记录猜测部分时间
auto guess_start = system_clock::now();

        while (!q.priority.empty())
        {
#ifdef TIME_COUNT
auto start_pop_next = system_clock::now();
#endif
            q.PopNext();
#ifdef TIME_COUNT
auto end_pop_next = system_clock::now();
auto duration_pop_next = duration_cast<microseconds>(end_pop_next - start_pop_next);
time_pop_next += double(duration_pop_next.count()) * microseconds::period::num / microseconds::period::den;
#endif
            //BUG 呃这里都直接赋值了， 那么guessing 里的 total_guesses+= 1 不是一点用没有？
#ifdef TIME_COUNT
auto start_check = system_clock::now();
#endif
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

// 结束猜测部分计时
auto guess_end = system_clock::now();
auto guess_duration = duration_cast<microseconds>(guess_end - guess_start);
time_guess_part += double(guess_duration.count()) * microseconds::period::num / microseconds::period::den;

                    cout << "\n--- 实验结果 ---" << endl;
                    cout << "猜测上限: " << GENERATE_N << ", 批处理大小: " << NUM_PER_HASH << endl;
                    cout << "Guesses generated: " << history + q.total_guesses << endl;
                    cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
                    cout << "Hash time: " << time_hash << " seconds" << endl;
                    cout << "Train time: " << time_train << " seconds" << endl;
                    cout << "Total time: " << time_guess << " seconds" << endl;
                    cout << "Real Guess time: " << time_guess_part << " seconds" << endl;

#ifdef TIME_COUNT
cout << "time all pop_next  :" << time_pop_next << endl;
cout << "time all check  :" << time_check << endl;
cout << "time gpu_kernel :" << time_gpu_kernel << endl ;
cout << "time_add_task:" << time_add_task << endl ;
cout << "time_launch_task:" << time_launch_task << endl ;
cout << "time_before_launch : " << time_before_launch  << endl ;
cout << "time_after_launch : " << time_after_launch  << endl ;
cout << "time_string_process: " << time_string_process << endl ;
cout << "time_memcpy_toh :" << time_memcpy_toh << endl ;

cout << "time_all_batch:" << time_all_batch << endl << endl ;

#endif
                    cout << "-------------------" << endl;
                    
                    break;
                }
            }
#ifdef TIME_COUNT
auto end_check = system_clock::now();
auto duration_check = duration_cast<microseconds>(end_check - start_check);
time_check += double(duration_check.count()) * microseconds::period::num / microseconds::period::den;
#endif

            //BUG 需要确保每次都大于
            // 达到批处理大小，进行哈希计算
            if (curr_num > NUM_PER_HASH)
            {
// 哈希开始，结束当前猜测部分计时
auto guess_end = system_clock::now();
auto guess_duration = duration_cast<microseconds>(guess_end - guess_start);
time_guess_part += double(guess_duration.count()) * microseconds::period::num / microseconds::period::den;

                auto start_hash = system_clock::now();
                
                #ifdef USING_SIMD

                                
                //TODO 用string_view 可能有问题。
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

                #else
                // 使用标准MD5计算
                bit32 state[4];
                
                for (string_view pw : q.guesses)
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
                delete[]h_guess_buffer;
                h_guess_buffer = nullptr;
// 哈希完成，重新开始计时新的猜测部分
guess_start = system_clock::now();

            }
        }

        // 如果还有任务，就清除...(不过我没用异步，所以不至于)
        task_manager->clean();
    }

    clean_gpu_ordered_values_data(gpu_data);
    
    cout << "\n--- 实验批次结束 ---\n" << endl;
    return 0;
}