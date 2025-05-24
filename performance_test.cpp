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

// 性能测试配置
struct PerformanceTest {
    int target_guesses;
    const char* description;
};

const PerformanceTest PERF_TESTS[] = {
    {1000000, "100万猜测"},
    {2000000, "200万猜测"},
    {3000000, "300万猜测"},
};

const int NUM_PERF_TESTS = sizeof(PERF_TESTS) / sizeof(PERF_TESTS[0]);

void run_sync_test(PriorityQueue& q, int target) {
    cout << "\n=== 同步模式测试 ===" << endl;
    
    q.init();
    q.guesses.clear();
    
    auto start = system_clock::now();
    
    while (!q.priority.empty() && q.guesses.size() < target) {
        q.PopNext();
        q.total_guesses = q.guesses.size();
    }
    
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    double time_taken = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    
    cout << "同步模式 - 生成 " << q.total_guesses << " 个猜测" << endl;
    cout << "耗时: " << time_taken << " 秒" << endl;
    cout << "速度: " << (q.total_guesses / time_taken) << " 猜测/秒" << endl;
}

void run_async_test(PriorityQueue& q, int target) {
    cout << "\n=== 异步模式测试 ===" << endl;
    
    q.init();
    q.guesses.clear();
    
    auto start = system_clock::now();
    
    while (!q.priority.empty() && q.guesses.size() < target) {
        // 异步模式下的PopNext已经包含了异步处理逻辑
        q.PopNext();
        q.total_guesses = q.guesses.size();
    }
    
    // 等待所有异步任务完成
    if (global_async_thread_pool) {
        global_async_thread_pool->wait_all_tasks();
    }
    
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    double time_taken = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    
    cout << "异步模式 - 生成 " << q.total_guesses << " 个猜测" << endl;
    cout << "耗时: " << time_taken << " 秒" << endl;
    cout << "速度: " << (q.total_guesses / time_taken) << " 猜测/秒" << endl;
}

int main()
{
    cout << "=== PCFG 异步线程池性能对比测试 ===" << endl;
    
    // 初始化线程池
    init_thread_pool();
    cout << "线程池初始化完成，使用 " << global_async_thread_pool->get_pool_size() << " 个线程" << endl;
    
    // 训练模型
    PriorityQueue q;
    cout << "正在训练模型..." << endl;
    
#ifdef _WIN32
    q.m.train(".\\guessdata\\small_Rockyou-singleLined-full.txt");
#else
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
#endif
    
    q.m.order();
    cout << "模型训练完成" << endl;
    
    // 运行性能测试
    for (int i = 0; i < NUM_PERF_TESTS; i++) {
        cout << "\n" << string(50, '=') << endl;
        cout << "测试 #" << (i+1) << ": " << PERF_TESTS[i].description << endl;
        cout << "目标猜测数: " << PERF_TESTS[i].target_guesses << endl;
        cout << string(50, '=') << endl;
        
        // 异步测试
        run_async_test(q, PERF_TESTS[i].target_guesses);
        
        cout << "\n测试完成" << endl;
        
        // 在测试之间清理一下
        cleanup_completed_task_groups();
        usleep(100000); // 100ms休息
    }
    
    cout << "\n=== 所有性能测试完成 ===" << endl;
    
    // 清理
    cleanup_thread_pool();
    
    return 0;
}
