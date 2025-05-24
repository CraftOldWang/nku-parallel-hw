#include "ThreadPoolAsync_fixed.h"

// 全局实例定义
AsyncThreadPoolFixed* global_async_thread_pool_fixed = nullptr;

// 初始化线程池
void init_async_thread_pool_fixed() {
    if (!global_async_thread_pool_fixed) {
        try {
            global_async_thread_pool_fixed = new AsyncThreadPoolFixed();
        } catch (...) {
            global_async_thread_pool_fixed = nullptr;
            throw;
        }
    }
}

// 清理线程池
void cleanup_async_thread_pool_fixed() {
    if (global_async_thread_pool_fixed) {
        delete global_async_thread_pool_fixed;
        global_async_thread_pool_fixed = nullptr;
    }
}
