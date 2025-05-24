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

// 逐步增加的实验配置，从最小开始测试
const ExperimentConfig EXPERIMENTS[] = {
    {1000000, 500000, "数据集/1M测试"},
    {2000000, 500000, "数据集/2M测试"},
    {3000000, 500000, "数据集/3M测试"},
    {5000000, 1000000, "数据集/5M测试"},
    {10000000, 1000000, "数据集/10M测试"},
};

// 要运行的实验数量
const int NUM_EXPERIMENTS = sizeof(EXPERIMENTS) / sizeof(EXPERIMENTS[0]);

int main()
{
    cout << "=== 开始异步线程池调试版本 ===" << endl;
    cout.flush();
    
    try {
        // 初始化线程池
        cout << "步骤1: 初始化线程池..." << endl;
        cout.flush();
        init_thread_pool();
        cout << "线程池初始化完成" << endl;
        cout.flush();
        
        // 添加时间戳
        auto now = system_clock::now();
        auto now_time = system_clock::to_time_t(now);
        auto now_tm = localtime(&now_time);

        cout << "\n--- WIN 异步线程池 MD5 实验批次 [";
        cout << put_time(now_tm, "%Y-%m-%d %H:%M:%S") << "] ---\n";
        cout.flush();
        
        // 训练模型（只需一次）
        cout << "步骤2: 开始训练模型..." << endl;
        cout.flush();
        PriorityQueue q;
        auto start_train = system_clock::now();
        
#ifdef _WIN32
        cout << "使用训练数据: .\\guessdata\\Rockyou-singleLined-full.txt" << endl;
        q.m.train(".\\guessdata\\Rockyou-singleLined-full.txt");
#else
        cout << "使用训练数据: /guessdata/Rockyou-singleLined-full.txt" << endl;
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
#endif
        
        cout << "训练数据加载完成，开始排序..." << endl;
        cout.flush();
        q.m.order();
        
        auto end_train = system_clock::now();
        auto duration_train = duration_cast<microseconds>(end_train - start_train);
        double time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
        
        cout << "模型训练完成，耗时: " << time_train << " 秒" << endl;
        if (global_async_thread_pool) {
            cout << "异步线程池大小: " << global_async_thread_pool->get_pool_size() << " 个线程" << endl;
        } else {
            cout << "警告: 异步线程池未初始化！" << endl;
        }
        cout.flush();
        
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
            cout.flush();
            
            try {
                cout << "步骤3: 重置队列..." << endl;
                cout.flush();
                
                // 重置队列
                q.init();
                q.guesses.clear();
                cout << "队列重置完成，优先队列大小: " << q.priority.size() << endl;
                cout.flush();
                
                double time_hash = 0;  // 用于MD5哈希的时间
                double time_guess = 0; // 哈希和猜测的总时长
                
                int curr_num = 0;
                auto start = system_clock::now();
                int history = 0;
                int iterations = 0;
                
                cout << "步骤4: 开始主循环..." << endl;
                cout.flush();
                
                while (!q.priority.empty())
                {
                    iterations++;
                    
                    // 更频繁的调试输出
                    if (iterations % 100 == 0) {
                        cout << "迭代 " << iterations << ": 猜测数=" << q.total_guesses 
                             << ", 活动组=" << get_active_task_groups_count();
                        if (global_async_thread_pool) {
                            cout << ", 队列=" << global_async_thread_pool->get_queue_size();
                        }
                        cout << ", 优先队列剩余=" << q.priority.size() << endl;
                        cout.flush();
                    }
                    
                    try {
                        // 检查异步任务状态
                        if (global_async_thread_pool) {
                            int active_groups = get_active_task_groups_count();
                            int queue_size = global_async_thread_pool->get_queue_size();
                            
                            // 更保守的控制策略
                            if (active_groups > 8 || queue_size > 100) {
                                if (iterations % 100 == 0) {
                                    cout << "等待任务: 活动组=" << active_groups << ", 队列=" << queue_size << endl;
                                    cout.flush();
                                }
                                usleep(5000); // 5ms等待
                                cleanup_completed_task_groups();
                            }
                        }
                        
                        // 执行PopNext前的状态检查
                        if (iterations % 500 == 0) {
                            cout << "执行PopNext前: 优先队列大小=" << q.priority.size() << endl;
                            cout.flush();
                        }
                        
                        q.PopNext();
                        q.total_guesses = q.guesses.size();
                        
                        // 执行PopNext后的状态检查
                        if (iterations % 500 == 0) {
                            cout << "执行PopNext后: 猜测数=" << q.total_guesses << ", 优先队列大小=" << q.priority.size() << endl;
                            cout.flush();
                        }
                        
                        // 检查内存使用情况
                        if (q.total_guesses > 5000000) {  // 降低到500万
                            cout << "内存保护：强制清理猜测缓存，当前猜测数=" << q.total_guesses << endl;
                            cout.flush();
                            q.guesses.clear();
                            q.total_guesses = q.guesses.size();
                        }
                        
                    } catch (const exception& e) {
                        cout << "PopNext 执行出错 (迭代 " << iterations << "): " << e.what() << endl;
                        cout.flush();
                        break;
                    } catch (...) {
                        cout << "PopNext 执行出现未知错误 (迭代 " << iterations << ")" << endl;
                        cout.flush();
                        break;
                    }
                    
                    if (q.total_guesses - curr_num >= 100000)
                    {
                        cout << "进度更新: 总猜测=" << (history + q.total_guesses) << ", 目标=" << GENERATE_N << endl;
                        cout.flush();
                        curr_num = q.total_guesses;
                        
                        // 检查是否达到当前实验的猜测上限
                        if (history + q.total_guesses > GENERATE_N)
                        {
                            cout << "达到目标，准备结束..." << endl;
                            cout.flush();
                            
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
                            cout << "迭代次数: " << iterations << endl;
                            cout << "-------------------" << endl;
                            cout.flush();
                            
                            break;
                        }
                    }
                    
                    // 达到批处理大小，进行哈希计算
                    if (curr_num > NUM_PER_HASH)
                    {
                        cout << "开始MD5哈希计算，猜测数=" << curr_num << endl;
                        cout.flush();
                        
                        auto start_hash = system_clock::now();
                        bit32 state[4];
                        
                        for (const string& pw : q.guesses)
                        {
                            MD5Hash(pw, state);
                        }
                        
                        // 计算哈希时间
                        auto end_hash = system_clock::now();
                        auto duration = duration_cast<microseconds>(end_hash - start_hash);
                        time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
                        
                        cout << "MD5计算完成，耗时=" << (time_hash * 1000) << "ms" << endl;
                        cout.flush();
                        
                        // 记录已经生成的口令总数
                        history += curr_num;
                        curr_num = 0;
                        q.guesses.clear();
                        
                        cout << "历史记录更新: history=" << history << endl;
                        cout.flush();
                    }
                    
                    // 防止无限循环
                    if (iterations > 1000000) {
                        cout << "达到最大迭代次数限制，强制退出" << endl;
                        cout.flush();
                        break;
                    }
                }
                
                cout << "主循环结束，最终迭代次数=" << iterations << endl;
                cout.flush();
                
            } catch (const exception& e) {
                cout << "实验 #" << (exp_idx + 1) << " 执行出错: " << e.what() << endl;
                cout.flush();
            } catch (...) {
                cout << "实验 #" << (exp_idx + 1) << " 执行出现未知错误" << endl;
                cout.flush();
            }
        }
        
        cout << "\n--- 异步线程池实验批次结束 ---\n" << endl;
        cout.flush();
        
        // 等待所有异步任务完成
        if (global_async_thread_pool) {
            cout << "等待所有异步任务完成..." << endl;
            cout.flush();
            global_async_thread_pool->wait_all_tasks();
            cout << "异步任务全部完成！" << endl;
            cout.flush();
        }
        
        // 清理线程池
        cout << "清理线程池..." << endl;
        cout.flush();
        cleanup_thread_pool();
        cout << "清理完成" << endl;
        cout.flush();
        
    } catch (const exception& e) {
        cout << "程序执行出错: " << e.what() << endl;
        cout.flush();
        return 1;
    } catch (...) {
        cout << "程序执行出现未知错误" << endl;
        cout.flush();
        return 1;
    }
    
    cout << "程序正常结束" << endl;
    cout.flush();
    return 0;
}
