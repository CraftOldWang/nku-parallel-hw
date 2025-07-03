#ifndef CONFIG_H
#define CONFIG_H

// #define NOT_USING_STRING_ARR // 定义宏，或者注释掉以禁用
// #define USING_ALIGNED


#define USING_SIMD // 定义宏以启用SIMD MD5计算

// #define USING_MPI // 定义宏以启用MPI并行处理

#define USING_SMALL

#define USING_POOL
// #define THREAD_NUM 8 // 线程池的线程数

// 每 10_0000 个guess 拿去给gpu处理一下
// #define GPU_BATCH_SIZE 100000
// 看看batch 在 10_0000  100_0000 1000_0000 有什么区别。 （guess time 会变吗）
#define DEBUG

// #define TIME_COUNT
#endif