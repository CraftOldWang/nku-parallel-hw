#include "ThreadPoolAsync.h"

// 全局异步线程池实例
AsyncThreadPool* global_async_thread_pool = nullptr;

// 初始化异步线程池
void init_async_thread_pool() {
    if (global_async_thread_pool == nullptr) {
        global_async_thread_pool = new AsyncThreadPool();
    }
}

// 清理异步线程池
void cleanup_async_thread_pool() {
    if (global_async_thread_pool != nullptr) {
        delete global_async_thread_pool;
        global_async_thread_pool = nullptr;
    }
}
