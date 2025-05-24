#include "PCFG.h"
#include "ThreadPool.h"
#include "ThreadPoolAsync.h"
#include <chrono>
#include "md5.h"
#include <iomanip>
using namespace std;
using namespace chrono;

// 声明异步任务管理函数
extern void cleanup_completed_task_groups();
extern int get_active_task_groups_count();

// 实验配置数组，每一组包含两个参数：总生成数量和批处理大小
struct ExperimentConfig {
    int generate_n;     // 猜测上限
    int batch_size;     // 一次处理的口令数量
    const char* label;  // 实验标签
};

// 在这里定义所有要运行的实验
const ExperimentConfig EXPERIMENTS[] = {
    {1000000, 1000000, "数据集/小批次"},
    {1500000, 1000000, "数据集/小批次"},
    {2000000, 1000000, "数据集/小批次"},
    {2500000, 1000000, "数据集/小批次"},
    {3000000, 1000000, "数据集/小批次"},
    {3500000, 1000000, "数据集/小批次"},
    {4000000, 1000000, "数据集/小批次"},
    {4500000, 1000000, "数据集/小批次"},
    {5000000, 1000000, "数据集/小批次"},
};

// 要运行的实验数量
const int NUM_EXPERIMENTS = sizeof(EXPERIMENTS) / sizeof(EXPERIMENTS[0]);

// 编译指令如下
// g++ main_pool.cpp train.cpp guessing_pthread_pool.cpp md5.cpp -o main_pool -lpthread
// g++ main_pool.cpp train.cpp guessing_pthread_pool.cpp md5.cpp -o main_pool -lpthread -O1
// g++ main_pool.cpp train.cpp guessing_pthread_pool.cpp md5.cpp -o main_pool -lpthread -O2

int main()
{
    // 初始化线程池
    init_thread_pool();
      // 添加时间戳
    auto now = system_clock::now();
    auto now_time = system_clock::to_time_t(now);
    auto now_tm = localtime(&now_time);

#ifdef _WIN32
    cout << "\n--- WIN 线程池 MD5 实验批次 [";
    cout << put_time(now_tm, "%Y-%m-%d %H:%M:%S") << "] ---\n";
#else
    cout << "\n--- 线程池 MD5 实验批次 [";
    cout << put_time(now_tm, "%Y-%m-%d %H:%M:%S") << "] ---\n";
#endif
    
    // 训练模型（只需一次）
    PriorityQueue q;
    auto start_train = system_clock::now();
// 将windows下用的main.cpp合并进来了
#ifdef _WIN32
    q.m.train(".\\guessdata\\Rockyou-singleLined-full.txt");
#else
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
#endif
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    double time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
    
    cout << "模型训练完成，耗时: " << time_train << " 秒" << endl;
    cout << "线程池大小: " << global_thread_pool->get_pool_size() << " 个线程" << endl;
    
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
        int history = 0;        while (!q.priority.empty())
        {
            try {
                // 检查异步任务状态，避免任务积压过多
                if (global_async_thread_pool) {
                    int active_groups = get_active_task_groups_count();
                    int queue_size = global_async_thread_pool->get_queue_size();
                    
                    // 如果任务积压太多，稍微等待一下
                    if (active_groups > 32 || queue_size > 500) {  // 降低阈值
                        usleep(5000); // 增加等待时间到5ms
                        cleanup_completed_task_groups();
                    }
                }
                
                // cout <<" here , can i pop?"<<endl;
                q.PopNext();
                // cout <<" yes i can"<<endl;
                q.total_guesses = q.guesses.size();
                
                // 检查内存使用情况，防止内存不足
                if (q.total_guesses > 50000000) {  // 如果猜测数量超过5000万，强制清理
                    cout << "内存保护：强制清理猜测缓存..." << endl;
                    q.guesses.clear();
                    q.total_guesses = q.guesses.size();
                }
                
            } catch (const exception& e) {
                cout << "PopNext 执行出错: " << e.what() << endl;
                break;
            } catch (...) {
                cout << "PopNext 执行出现未知错误" << endl;
                break;
            }
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
                bit32 state[4];
                
                for (string pw : q.guesses)
                {
                    MD5Hash(pw, state);
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
      cout << "\n--- 线程池实验批次结束 ---\n" << endl;
    
    // 等待所有异步任务完成
    extern AsyncThreadPool* global_async_thread_pool;
    if (global_async_thread_pool) {
        cout << "等待所有异步任务完成..." << endl;
        global_async_thread_pool->wait_all_tasks();
        cout << "异步任务全部完成！" << endl;
    }
    
    // 清理线程池
    cleanup_thread_pool();
    
    return 0;
}
