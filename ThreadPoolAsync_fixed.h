#ifndef THREADPOOLASYNC_FIXED_H
#define THREADPOOLASYNC_FIXED_H

#include <pthread.h>
#include <queue>
#include <vector>
#include <string>
#include <map>
#include <unistd.h>
#include "PCFG.h"

using namespace std;

#define POOL_SIZE_TOUSE 16  // 减少线程池大小以提高稳定性

// 任务结构体
struct AsyncTaskFixed {
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

// 修复版异步线程池类
class AsyncThreadPoolFixed {
private:
    static const int POOL_SIZE = POOL_SIZE_TOUSE;  // 线程池大小
    pthread_t threads[POOL_SIZE];    // 线程数组
    queue<AsyncTaskFixed> task_queue; // 任务队列
    pthread_mutex_t queue_mutex;     // 队列互斥锁
    pthread_cond_t condition;        // 条件变量
    pthread_cond_t finished_condition; // 完成条件变量
    bool stop_flag;                  // 停止标志
    int active_tasks;                // 活跃任务数
    int pending_tasks;               // 待处理任务数
    int current_group_id;            // 当前任务组ID
    
    // 任务组状态跟踪（改进版）
    struct TaskGroupFixed {
        int total_tasks;
        int completed_tasks;
        bool is_active;
        pthread_mutex_t group_mutex;  // 每个组有自己的锁
        
        TaskGroupFixed() : total_tasks(0), completed_tasks(0), is_active(false) {
            pthread_mutex_init(&group_mutex, NULL);
        }
        
        ~TaskGroupFixed() {
            pthread_mutex_destroy(&group_mutex);
        }
    };
    
    map<int, TaskGroupFixed*> task_groups; // 跟踪每个任务组的状态
    pthread_mutex_t groups_mutex;          // 保护task_groups的锁
    
    // 最大队列大小限制，防止内存过度使用
    static const int MAX_QUEUE_SIZE = 1000;
    
    // 工作线程函数
    static void* worker_thread(void* arg) {
        AsyncThreadPoolFixed* pool = (AsyncThreadPoolFixed*)arg;
        
        while (true) {
            AsyncTaskFixed task;
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
                try {
                    pool->execute_task(task);
                    
                    // 标记任务完成
                    pthread_mutex_lock(&pool->queue_mutex);
                    pool->active_tasks--;
                    
                    // 更新任务组状态（安全方式）
                    pthread_mutex_lock(&pool->groups_mutex);
                    auto it = pool->task_groups.find(task.task_group_id);
                    if (it != pool->task_groups.end() && it->second) {
                        pthread_mutex_lock(&it->second->group_mutex);
                        it->second->completed_tasks++;
                        pthread_mutex_unlock(&it->second->group_mutex);
                    }
                    pthread_mutex_unlock(&pool->groups_mutex);
                    
                    // 如果所有任务完成或者没有待处理任务，通知等待线程
                    if (pool->active_tasks == 0 && pool->pending_tasks == 0) {
                        pthread_cond_broadcast(&pool->finished_condition);
                    }
                    
                    pthread_mutex_unlock(&pool->queue_mutex);
                } catch (...) {
                    // 处理任务执行异常
                    pthread_mutex_lock(&pool->queue_mutex);
                    pool->active_tasks--;
                    pthread_mutex_unlock(&pool->queue_mutex);
                }
            }
        }
        return NULL;
    }
    
    // 执行具体任务（增强错误处理）
    void execute_task(const AsyncTaskFixed& task) {
        if (!task.seg || !task.result_vec || !task.counter || !task.mutex) {
            return; // 安全检查
        }
        
        vector<string> local_guesses;
        int local_count = 0;
        
        // 预分配空间以提高性能
        int expected_size = task.end_idx - task.start_idx;
        if (expected_size > 0 && expected_size < 1000000) { // 防止过大分配
            local_guesses.reserve(expected_size);
        }
        
        // 生成guess字符串
        try {
            for (int i = task.start_idx; i < task.end_idx; i++) {
                if (i >= 0 && i < (int)task.seg->ordered_values.size()) {
                    if (task.is_single_segment) {
                        local_guesses.push_back(task.seg->ordered_values[i]);
                    } else {
                        local_guesses.push_back(task.prefix + task.seg->ordered_values[i]);
                    }
                    local_count++;
                }
            }
            
            // 将结果合并到全局容器中
            pthread_mutex_lock(task.mutex);
            task.result_vec->insert(task.result_vec->end(), local_guesses.begin(), local_guesses.end());
            *(task.counter) += local_count;
            pthread_mutex_unlock(task.mutex);
        } catch (...) {
            // 确保在异常情况下也能释放锁
            pthread_mutex_unlock(task.mutex);
            throw;
        }
    }
    
public:
    // 构造函数
    AsyncThreadPoolFixed() : stop_flag(false), active_tasks(0), pending_tasks(0), current_group_id(0) {
        pthread_mutex_init(&queue_mutex, NULL);
        pthread_mutex_init(&groups_mutex, NULL);
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
    ~AsyncThreadPoolFixed() {
        // 设置停止标志并唤醒所有线程
        pthread_mutex_lock(&queue_mutex);
        stop_flag = true;
        pthread_cond_broadcast(&condition);
        pthread_mutex_unlock(&queue_mutex);
        
        // 等待所有线程结束
        for (int i = 0; i < POOL_SIZE; i++) {
            pthread_join(threads[i], NULL);
        }
        
        // 清理任务组
        pthread_mutex_lock(&groups_mutex);
        for (auto& pair : task_groups) {
            delete pair.second;
        }
        task_groups.clear();
        pthread_mutex_unlock(&groups_mutex);
        
        // 清理资源
        pthread_mutex_destroy(&queue_mutex);
        pthread_mutex_destroy(&groups_mutex);
        pthread_cond_destroy(&condition);
        pthread_cond_destroy(&finished_condition);
    }
    
    // 异步提交任务（改进版）
    int submit_tasks_async(const vector<AsyncTaskFixed>& tasks) {
        if (tasks.empty()) {
            return -1;
        }
        
        pthread_mutex_lock(&queue_mutex);
        
        // 检查队列大小，防止内存过度使用
        if (task_queue.size() + tasks.size() > MAX_QUEUE_SIZE) {
            pthread_mutex_unlock(&queue_mutex);
            return -1; // 拒绝提交过多任务
        }
        
        // 生成新的组ID
        int group_id = ++current_group_id;
        pthread_mutex_unlock(&queue_mutex);
        
        // 创建任务组
        pthread_mutex_lock(&groups_mutex);
        TaskGroupFixed* group = new TaskGroupFixed();
        group->total_tasks = tasks.size();
        group->completed_tasks = 0;
        group->is_active = true;
        task_groups[group_id] = group;
        pthread_mutex_unlock(&groups_mutex);
        
        // 提交任务
        pthread_mutex_lock(&queue_mutex);
        for (AsyncTaskFixed task : tasks) {
            task.task_group_id = group_id;
            task_queue.push(task);
            pending_tasks++;
        }
        pthread_cond_broadcast(&condition);
        pthread_mutex_unlock(&queue_mutex);
        
        return group_id;
    }
    
    // 检查任务组是否完成
    bool is_group_completed(int group_id) {
        pthread_mutex_lock(&groups_mutex);
        auto it = task_groups.find(group_id);
        if (it == task_groups.end() || !it->second) {
            pthread_mutex_unlock(&groups_mutex);
            return true; // 不存在的组认为已完成
        }
        
        TaskGroupFixed* group = it->second;
        pthread_mutex_lock(&group->group_mutex);
        bool completed = (group->completed_tasks >= group->total_tasks);
        pthread_mutex_unlock(&group->group_mutex);
        pthread_mutex_unlock(&groups_mutex);
        
        return completed;
    }
    
    // 清理已完成的任务组
    void cleanup_completed_groups() {
        vector<int> completed_groups;
        
        pthread_mutex_lock(&groups_mutex);
        auto it = task_groups.begin();
        while (it != task_groups.end()) {
            if (it->second) {
                pthread_mutex_lock(&it->second->group_mutex);
                bool completed = (it->second->completed_tasks >= it->second->total_tasks);
                pthread_mutex_unlock(&it->second->group_mutex);
                
                if (completed) {
                    completed_groups.push_back(it->first);
                    delete it->second;
                    it = task_groups.erase(it);
                } else {
                    ++it;
                }
            } else {
                it = task_groups.erase(it);
            }
        }
        pthread_mutex_unlock(&groups_mutex);
    }
    
    // 获取活动任务组数量
    int get_active_groups_count() {
        pthread_mutex_lock(&groups_mutex);
        int count = task_groups.size();
        pthread_mutex_unlock(&groups_mutex);
        return count;
    }
    
    // 等待所有任务完成
    void wait_all_tasks() {
        pthread_mutex_lock(&queue_mutex);
        while (active_tasks > 0 || pending_tasks > 0) {
            pthread_cond_wait(&finished_condition, &queue_mutex);
        }
        pthread_mutex_unlock(&queue_mutex);
        
        // 清理所有任务组
        cleanup_completed_groups();
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
    
    // 获取统计信息
    void get_stats(int& active, int& pending, int& groups) {
        pthread_mutex_lock(&queue_mutex);
        active = active_tasks;
        pending = pending_tasks;
        pthread_mutex_unlock(&queue_mutex);
        
        pthread_mutex_lock(&groups_mutex);
        groups = task_groups.size();
        pthread_mutex_unlock(&groups_mutex);
    }
};

// 全局实例声明
extern AsyncThreadPoolFixed* global_async_thread_pool_fixed;

// 管理函数
void init_async_thread_pool_fixed();
void cleanup_async_thread_pool_fixed();

#endif // THREADPOOLASYNC_FIXED_H
