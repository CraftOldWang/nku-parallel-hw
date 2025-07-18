#ifndef CONFIG_H
#define CONFIG_H

// #define NOT_USING_STRING_ARR // 定义宏，或者注释掉以禁用
// #define USING_ALIGNED

#ifndef USING_SIMD
#define USING_SIMD // 定义宏以启用SIMD MD5计算
#endif
// #define USING_MPI // 定义宏以启用MPI并行处理

// #define USING_SMALL
#ifndef USING_POOL
#define USING_POOL
#endif

// #ifndef THREAD_NUM
// #define THREAD_NUM 8 // 线程池的线程数 , 是脚本的参数， 所以其实这里没有用
// #endif

#ifndef HASH_THREAD_NUM
#define HASH_THREAD_NUM 32  // Hash线程池大小，可以独立调整  不作为脚本的参数.... 
#endif

// CUDA GPU parameters
#ifndef GUESS_PER_THREAD
#define GUESS_PER_THREAD 8  // Default value if not defined at compile time
#endif



// 每 10_0000 个guess 拿去给gpu处理一下
// #define GPU_BATCH_SIZE 100000
// 看看batch 在 10_0000  100_0000 1000_0000 有什么区别。 （guess time 会变吗）
// #define DEBUG
// 编译命令
// chcp 65001 && & "D:\Softwares\Git\bin\bash.exe" "run_cuda.sh"

// #define TIME_COUNT
#endif