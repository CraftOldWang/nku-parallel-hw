#include "PCFG.h"
#include "ThreadPool.h"
#include "ThreadPoolAsync.h"
#include <chrono>
#include "md5.h"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace chrono;

// 声明异步任务管理函数
extern void cleanup_completed_task_groups();
extern int get_active_task_groups_count();

int main()
{
    try {
        cout << "=== Async Thread Pool Performance Test ===" << endl;
        
        // 初始化线程池
        cout << "Initializing thread pools..." << endl;
        init_thread_pool();
        
        if (!global_thread_pool || !global_async_thread_pool) {
            cout << "Error: Failed to initialize thread pools!" << endl;
            return 1;
        }
        
        cout << "Thread pool size: " << global_thread_pool->get_pool_size() << endl;
        cout << "Async thread pool size: " << global_async_thread_pool->get_pool_size() << endl;
        
        // 训练模型
        PriorityQueue q;
        auto start_train = system_clock::now();
        
        cout << "Loading training data..." << endl;
#ifdef _WIN32
        q.m.train(".\\guessdata\\small_Rockyou-singleLined-full.txt");
#else
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
#endif
        
        q.m.order();
        auto end_train = system_clock::now();
        auto duration_train = duration_cast<microseconds>(end_train - start_train);
        double time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
        
        cout << "Model training completed, time: " << time_train << " seconds" << endl;
        
        // 简化的测试配置
        int GENERATE_N = 1000000;  // 减少到100万
        int NUM_PER_HASH = 100000; // 减少批处理大小
        
        cout << "Starting test with limit: " << GENERATE_N << ", batch size: " << NUM_PER_HASH << endl;
        
        // 初始化队列
        q.init();
        q.guesses.clear();
        
        cout << "Queue initialized with " << q.priority.size() << " PTs." << endl;
        
        double time_hash = 0;
        double time_guess = 0;
        
        int curr_num = 0;
        auto start = system_clock::now();
        int history = 0;
        int iterations = 0;
        
        while (!q.priority.empty() && iterations < 100) { // 限制迭代次数
            iterations++;
            
            // 检查异步任务状态
            int active_groups = get_active_task_groups_count();
            int queue_size = 0;
            if (global_async_thread_pool) {
                queue_size = global_async_thread_pool->get_queue_size();
            }
            
            if (iterations % 10 == 0) {
                cout << "Iteration " << iterations << ": Active groups=" << active_groups 
                     << ", Queue size=" << queue_size << ", Guesses=" << q.guesses.size() << endl;
            }
            
            // 如果任务积压太多，等待一下
            if (active_groups > 64 || queue_size > 1000) {
                usleep(1000);
                cleanup_completed_task_groups();
            }
            
            q.PopNext();
            q.total_guesses = q.guesses.size();
            
            // 检查是否达到猜测上限
            if (history + q.total_guesses > GENERATE_N) {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                
                cout << "\n=== Test Results ===" << endl;
                cout << "Target limit: " << GENERATE_N << endl;
                cout << "Guesses generated: " << history + q.total_guesses << endl;
                cout << "Iterations: " << iterations << endl;
                cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
                cout << "Hash time: " << time_hash << " seconds" << endl;
                cout << "Train time: " << time_train << " seconds" << endl;
                cout << "Total time: " << time_guess << " seconds" << endl;
                break;
            }
            
            // 达到批处理大小，进行哈希计算
            if (q.total_guesses - curr_num >= NUM_PER_HASH) {
                curr_num = q.total_guesses;
                
                auto start_hash = system_clock::now();
                bit32 state[4];
                
                for (const string& pw : q.guesses) {
                    MD5Hash(pw, state);
                }
                
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
                
                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
                
                cout << "Processed batch. Total processed: " << history << endl;
            }
        }
        
        cout << "\n=== Final Cleanup ===" << endl;
        cout << "Waiting for all async tasks to complete..." << endl;
        
        if (global_async_thread_pool) {
            global_async_thread_pool->wait_all_tasks();
            cout << "All async tasks completed!" << endl;
        }
        
        cleanup_thread_pool();
        cout << "Thread pools cleaned up." << endl;
        
    } catch (const exception& e) {
        cout << "Exception caught: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "Unknown exception caught!" << endl;
        return 1;
    }
    
    return 0;
}
