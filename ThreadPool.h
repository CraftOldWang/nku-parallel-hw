#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <atomic>  // 🔥 新增


class ThreadPool {
public:
    ThreadPool(size_t);
    template<class FUNCTION, class... Args>
    auto enqueue(FUNCTION&& f, Args&&... args) 
        -> std::future<typename std::result_of<FUNCTION(Args...)>::type>;
    ~ThreadPool();



    // 🔥 新增：获取监控信息
    size_t pending_tasks() const { return pending_task_count.load(); }
    size_t completed_tasks() const { return completed_task_count.load(); }
    size_t worker_count() const { return workers.size(); }


    // 🔥 新增：打印状态信息
    void print_status(const char* label = "") const {
        printf("[ThreadPool %s] Pending: %zu, Completed: %zu, Workers: %zu\n", 
               label, pending_tasks(), completed_tasks(), worker_count());
    }
private:
    // need to keep track of threads so we can join them
    std::vector< std::thread > workers;
    // the task queue
    std::queue< std::function<void()> > tasks;
    
    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;


    // 🔥 新增：原子计数器
    mutable std::atomic<size_t> pending_task_count{0};    // 等待执行的任务数
    mutable std::atomic<size_t> completed_task_count{0};  // 已完成的任务数

};
 
// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads)
    :   stop(false)
{
    for(size_t i = 0;i<threads;++i)
        workers.emplace_back(
            [this,  i]
            {
                for(;;)
                {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                            [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();

                        // 🔥 新增：任务开始执行，从等待队列移除
                        pending_task_count--;
#ifdef THREAD_POOL_DEBUG
                        printf("[ThreadPool] Worker-%zu picked up task. Pending: %zu, Queue size: %zu\n", 
                               i, pending_task_count.load(), this->tasks.size());
                        fflush(stdout);
#endif
                    }
                    // 🔥 新增：执行任务并计数
                    try {
                        task();
                        completed_task_count++;
#ifdef THREAD_POOL_DEBUG
                        printf("[ThreadPool] Task completed. Pending: %zu, Completed: %zu\n", 
                               pending_task_count.load(), completed_task_count.load());
                        fflush(stdout);
#endif
                    } catch (const std::exception& e) {
                        printf("[ThreadPool ERROR] Worker-%zu task failed: %s\n", i, e.what());
                        completed_task_count++;  // 失败也算完成
                    } catch (...) {
                        printf("[ThreadPool ERROR] Worker-%zu task failed with unknown exception\n", i);
                        completed_task_count++;  // 失败也算完成
                    }

                }
            }
        );
}

// add new work item to the pool
template<class FUNCTION, class... Args>
auto ThreadPool::enqueue(FUNCTION&& f, Args&&... args) 
    -> std::future<typename std::result_of<FUNCTION(Args...)>::type>
{
    using return_type = typename std::result_of<FUNCTION(Args...)>::type;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<FUNCTION>(f), std::forward<Args>(args)...)
        );
        
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // don't allow enqueueing after stopping the pool
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task](){ (*task)(); });

        // 🔥 新增：任务添加到等待队列
        pending_task_count++;
#ifdef THREAD_POOL_DEBUG
        // 🔥 新增：添加任务时的打印信息
        printf("[ThreadPool] Task enqueued. Pending: %zu, Queue size: %zu, Completed: %zu\n", 
               pending_task_count.load(), tasks.size(), completed_task_count.load());
        fflush(stdout);
#endif

    }
    condition.notify_one();
    return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
    // 🔥 新增：析构前打印最终状态
    printf("[ThreadPool] Shutting down. Final stats - Pending: %zu, Completed: %zu\n", 
           pending_task_count.load(), completed_task_count.load());

    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers)
        worker.join();

    // 🔥 新增：检查是否有遗留任务
    if (pending_task_count.load() > 0) {
        printf("[ThreadPool WARNING] %zu tasks were not completed!\n", pending_task_count.load());
    }

}

#endif