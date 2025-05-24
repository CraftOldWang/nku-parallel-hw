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
    cout << "=== 异步线程池测试程序 ===" << endl;
    
    try {
        // 初始化线程池
        cout << "正在初始化线程池..." << endl;
        init_thread_pool();
        cout << "线程池初始化完成" << endl;
        
        // 训练模型
        PriorityQueue q;
        cout << "正在加载和训练模型..." << endl;
        
#ifdef _WIN32
        q.m.train(".\\guessdata\\small_Rockyou-singleLined-full.txt");
#else
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
#endif
        
        q.m.order();
        cout << "模型训练完成" << endl;
        
        // 初始化队列
        cout << "正在初始化优先队列..." << endl;
        q.init();
        q.guesses.clear();
        cout << "队列初始化完成，包含 " << q.priority.size() << " 个 PTs" << endl;
        
        // 测试几次 PopNext 操作
        int test_count = 3;
        cout << "开始测试 " << test_count << " 次 PopNext 操作..." << endl;
        
        for (int i = 0; i < test_count && !q.priority.empty(); i++) {
            cout << "\n--- 测试轮次 " << (i+1) << " ---" << endl;
            cout << "当前活动任务组: " << get_active_task_groups_count() << endl;
            
            if (global_async_thread_pool) {
                cout << "异步队列大小: " << global_async_thread_pool->get_queue_size() << endl;
            }
            
            cout << "当前猜测数量: " << q.guesses.size() << endl;
            cout << "执行 PopNext..." << endl;
            
            // 执行 PopNext
            q.PopNext();
            q.total_guesses = q.guesses.size();
            
            cout << "PopNext 完成，新增猜测: " << q.total_guesses << endl;
            
            // 给异步任务一些时间执行
            cout << "等待异步任务执行..." << endl;
            usleep(50000); // 50ms
            cleanup_completed_task_groups();
            
            cout << "清理后活动任务组: " << get_active_task_groups_count() << endl;
        }
        
        cout << "\n=== 测试完成 ===" << endl;
        cout << "最终统计:" << endl;
        cout << "总猜测数: " << q.total_guesses << endl;
        cout << "剩余活动任务组: " << get_active_task_groups_count() << endl;
        
        // 等待所有异步任务完成
        if (global_async_thread_pool) {
            cout << "等待所有异步任务完成..." << endl;
            global_async_thread_pool->wait_all_tasks();
            cout << "所有异步任务已完成" << endl;
        }
        
        // 清理资源
        cout << "清理线程池..." << endl;
        cleanup_thread_pool();
        cout << "清理完成" << endl;
        
    } catch (const exception& e) {
        cout << "发生异常: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "发生未知异常" << endl;
        return 1;
    }
    
    cout << "程序正常结束" << endl;
    return 0;
}
