#ifndef THREADPOOLASYNC_H
#define THREADPOOLASYNC_H

#include <pthread.h>
#include <queue>
#include <vector>
#include <string>
#include <map>
#include <unistd.h>
#include "PCFG.h"

using namespace std;

#define POOL_SIZE_TOUSE 32  // 线程池大小

// 任务结构体
struct AsyncTask {
    segment* seg;                    // 指向segment数据的指针
    string prefix;                   // 前缀字符串（单segment时为空）
    int start_idx;                   // 处理的起始索引
    int end_idx;                     // 处理的结束索引
    vector<string>* result_vec;      // 结果存储容器
    int* counter;                    // 计数器
    pthread_mutex_t* mutex;          // 互斥锁
    bool is_single_segment;          // 是否为单segment处理
    int task_group_id;               // 任务组ID，用于跟踪哪些任务属于同一个Generate调用
};

// 异步线程池类
class AsyncThreadPool {
private:
    static const int POOL_SIZE = POOL_SIZE_TOUSE;  // 线程池大小
    pthread_t threads[POOL_SIZE];    // 线程数组
    queue<AsyncTask> task_queue;     // 任务队列
    pthread_mutex_t queue_mutex;     // 队列互斥锁
    pthread_cond_t condition;        // 条件变量
    pthread_cond_t finished_condition; // 完成条件变量
    bool stop_flag;                  // 停止标志
    int active_tasks;                // 活跃任务数
    int pending_tasks;               // 待处理任务数
    int current_group_id;            // 当前任务组ID
    
    // 任务组状态跟踪
    struct TaskGroup {
        int total_tasks;
        int completed_tasks;
        bool is_active;
    };
    
    map<int, TaskGroup> task_groups; // 跟踪每个任务组的状态
    
    // 工作线程函数
    static void* worker_thread(void* arg) {
        AsyncThreadPool* pool = (AsyncThreadPool*)arg;
        
        while (true) {
            AsyncTask task;
            bool has_task = false;
            
            // 获取任务
            pthread_mutex_lock(&pool->queue_mutex);
            while (pool->task_queue.empty() && !pool->stop_flag) {
                pthread_cond_wait(&pool->condition, &pool->queue_mutex);
            }
            
            if (pool->stop_flag && pool->task_queue.empty()) {
                pthread_mutex_unlock(&pool->queue_mutex);
                break;
            }
            
            if (!pool->task_queue.empty()) {
                task = pool->task_queue.front();
                pool->task_queue.pop();
                pool->pending_tasks--;
                pool->active_tasks++;
                has_task = true;
            }
            pthread_mutex_unlock(&pool->queue_mutex);
            
            // 执行任务
            if (has_task) {
                pool->execute_task(task);
                
                // 标记任务完成
                pthread_mutex_lock(&pool->queue_mutex);
                pool->active_tasks--;
                
                // 更新任务组状态
                auto& group = pool->task_groups[task.task_group_id];
                group.completed_tasks++;
                
                // 如果所有任务完成或者没有待处理任务，通知等待线程
                if (pool->active_tasks == 0 && pool->pending_tasks == 0) {
                    pthread_cond_broadcast(&pool->finished_condition);
                }
                
                pthread_mutex_unlock(&pool->queue_mutex);
            }
        }
        return NULL;
    }
    
    // 执行具体任务
    void execute_task(const AsyncTask& task) {
        vector<string> local_guesses;
        int local_count = 0;
        
        // 预分配空间以提高性能
        local_guesses.reserve(task.end_idx - task.start_idx);
        
        // 生成guess字符串
        for (int i = task.start_idx; i < task.end_idx; i++) {
            if (task.is_single_segment) {
                local_guesses.push_back(task.seg->ordered_values[i]);
            } else {
                local_guesses.push_back(task.prefix + task.seg->ordered_values[i]);
            }
            local_count++;
        }
        
        // 将结果合并到全局容器中
        pthread_mutex_lock(task.mutex);
        task.result_vec->insert(task.result_vec->end(), local_guesses.begin(), local_guesses.end());
        *(task.counter) += local_count;
        pthread_mutex_unlock(task.mutex);
    }
    
public:
    // 构造函数
    AsyncThreadPool() : stop_flag(false), active_tasks(0), pending_tasks(0), current_group_id(0) {
        pthread_mutex_init(&queue_mutex, NULL);
        pthread_cond_init(&condition, NULL);
        pthread_cond_init(&finished_condition, NULL);
        
        // 创建工作线程
        for (int i = 0; i < POOL_SIZE; i++) {
            if (pthread_create(&threads[i], NULL, worker_thread, this) != 0) {
                // 处理线程创建失败的情况
                stop_flag = true;
                throw "Failed to create worker thread";
            }
        }
    }
    
    // 析构函数
    ~AsyncThreadPool() {
        // 设置停止标志并唤醒所有线程
        pthread_mutex_lock(&queue_mutex);
        stop_flag = true;
        pthread_cond_broadcast(&condition);
        pthread_mutex_unlock(&queue_mutex);
        
        // 等待所有线程结束
        for (int i = 0; i < POOL_SIZE; i++) {
            pthread_join(threads[i], NULL);
        }
        
        // 清理资源
        pthread_mutex_destroy(&queue_mutex);
        pthread_cond_destroy(&condition);
        pthread_cond_destroy(&finished_condition);
    }
    
    // 异步提交任务到线程池（不等待完成）
    int submit_tasks_async(const vector<AsyncTask>& tasks) {
        pthread_mutex_lock(&queue_mutex);
        
        int group_id = ++current_group_id;
        
        // 初始化任务组
        TaskGroup group;
        group.total_tasks = tasks.size();
        group.completed_tasks = 0;
        group.is_active = true;
        task_groups[group_id] = group;
        
        // 添加任务到队列，设置组ID
        for (AsyncTask task : tasks) {
            task.task_group_id = group_id;
            task_queue.push(task);
            pending_tasks++;
        }
        
        pthread_cond_broadcast(&condition);
        pthread_mutex_unlock(&queue_mutex);
        
        return group_id; // 返回任务组ID，用于后续查询
    }
    
    // 检查指定任务组是否完成
    bool is_group_completed(int group_id) {
        pthread_mutex_lock(&queue_mutex);
        bool completed = false;
        
        auto it = task_groups.find(group_id);
        if (it != task_groups.end()) {
            completed = (it->second.completed_tasks >= it->second.total_tasks);
        }
        
        pthread_mutex_unlock(&queue_mutex);
        return completed;
    }
    
    // 等待指定任务组完成
    void wait_group(int group_id) {
        pthread_mutex_lock(&queue_mutex);
        
        while (task_groups.find(group_id) != task_groups.end() && 
               task_groups[group_id].completed_tasks < task_groups[group_id].total_tasks) {
            pthread_cond_wait(&finished_condition, &queue_mutex);
        }
        
        // 清理已完成的任务组
        task_groups.erase(group_id);
        
        pthread_mutex_unlock(&queue_mutex);
    }
    
    // 传统的同步提交（兼容现有代码）
    void submit_tasks(const vector<AsyncTask>& tasks) {
        int group_id = submit_tasks_async(tasks);
        wait_group(group_id);
    }
    
    // 等待所有任务完成
    void wait_all_tasks() {
        pthread_mutex_lock(&queue_mutex);
        while (active_tasks > 0 || pending_tasks > 0) {
            pthread_cond_wait(&finished_condition, &queue_mutex);
        }
        pthread_mutex_unlock(&queue_mutex);
    }
    
    // 获取线程池大小
    int get_pool_size() const {
        return POOL_SIZE;
    }
    
    // 获取当前队列中的任务数
    int get_queue_size() {
        pthread_mutex_lock(&queue_mutex);
        int size = task_queue.size();
        pthread_mutex_unlock(&queue_mutex);
        return size;
    }
    
    // 获取活跃任务组数量
    int get_active_groups() {
        pthread_mutex_lock(&queue_mutex);
        int count = task_groups.size();
        pthread_mutex_unlock(&queue_mutex);
        return count;
    }
};

// 全局异步线程池实例
extern AsyncThreadPool* global_async_thread_pool;

// 线程池管理函数
void init_async_thread_pool();
void cleanup_async_thread_pool();

#endif // THREADPOOLASYNC_H
