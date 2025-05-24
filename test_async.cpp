#include "PCFG.h"
#include "ThreadPool.h"
#include "ThreadPoolAsync.h"
#include <chrono>
#include "md5.h"
#include <iostream>
using namespace std;
using namespace chrono;

// 声明异步任务管理函数
extern void cleanup_completed_task_groups();
extern int get_active_task_groups_count();

int main()
{
    cout << "Testing async thread pool..." << endl;
    
    // 初始化线程池
    init_thread_pool();
    cout << "Thread pools initialized." << endl;
    
    // 训练模型
    PriorityQueue q;
    cout << "Loading model..." << endl;
    
#ifdef _WIN32
    q.m.train(".\\guessdata\\small_Rockyou-singleLined-full.txt");
#else
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
#endif
    
    q.m.order();
    cout << "Model loaded and ordered." << endl;
    
    // 初始化队列
    q.init();
    q.guesses.clear();
    cout << "Queue initialized with " << q.priority.size() << " PTs." << endl;
    
    // 测试少量的PopNext操作
    int test_iterations = 5;
    for (int i = 0; i < test_iterations && !q.priority.empty(); i++) {
        cout << "Iteration " << (i+1) << ": ";
        cout << "Active groups: " << get_active_task_groups_count() << ", ";
        if (global_async_thread_pool) {
            cout << "Queue size: " << global_async_thread_pool->get_queue_size() << ", ";
        }
        cout << "Guesses: " << q.guesses.size() << endl;
        
        q.PopNext();
        q.total_guesses = q.guesses.size();
        
        // 小的延迟让异步任务有时间执行
        usleep(10000); // 10ms
        cleanup_completed_task_groups();
    }
    
    cout << "Test completed. Final stats:" << endl;
    cout << "Total guesses: " << q.total_guesses << endl;
    cout << "Active groups: " << get_active_task_groups_count() << endl;
    
    // 等待所有任务完成
    if (global_async_thread_pool) {
        cout << "Waiting for all async tasks to complete..." << endl;
        global_async_thread_pool->wait_all_tasks();
        cout << "All async tasks completed!" << endl;
    }
    
    // 清理
    cleanup_thread_pool();
    cout << "Thread pools cleaned up." << endl;
    
    return 0;
}
