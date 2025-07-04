cmake_minimum_required(VERSION 3.20)
project(GuessHashCUDA LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_COMPILER "D:/Softwares/CUDA/v12.9/bin/nvcc.exe")

# 设置默认构建类型为 Release（优化）
#if(NOT CMAKE_BUILD_TYPE)
#  set(CMAKE_BUILD_TYPE Release)
#endif()


# 查找 CUDA Toolkit
# 这会设置 CUDAToolkit_INCLUDE_DIRS 等变量，用于后续引用 CUDA 头文件
find_package(CUDAToolkit REQUIRED)


# 所有源码文件（都在根目录）
set(SOURCES
    ${CMAKE_SOURCE_DIR}/correctness_guess.cpp
    ${CMAKE_SOURCE_DIR}/guessing.cpp
    ${CMAKE_SOURCE_DIR}/guessing_cuda.cu
    ${CMAKE_SOURCE_DIR}/train.cpp
    ${CMAKE_SOURCE_DIR}/md5.cpp
    ${CMAKE_SOURCE_DIR}/md5_avx.cpp
)

# 构建配置：GPU_BATCH_SIZE=1000000, GUESS_PER_THREAD=16, THREAD_NUM=8
set(BATCH_SIZE 1000000)
set(GUESS_PER_THREAD 16)
set(THREAD_NUM 8)

# 输出文件名
set(TARGET_NAME guess_correct_test_bs${BATCH_SIZE}_gpt${GUESS_PER_THREAD}_trn${THREAD_NUM})

add_executable(${TARGET_NAME} ${SOURCES})

# 定义编译宏
target_compile_definitions(${TARGET_NAME} PRIVATE
    GPU_BATCH_SIZE=${BATCH_SIZE}
    GUESS_PER_THREAD=${GUESS_PER_THREAD}
    THREAD_NUM=${THREAD_NUM}
    USING_POOL
    USING_SIMD
    NDEBUG
)

# 加入 CUDA include 路径，确保 device_launch_parameters.h 能找到
#target_include_directories(${TARGET_NAME} PRIVATE
#        ${CUDA_TOOLKIT_ROOT_DIR}/include
#)

# 加入 CUDA include 路径，确保 C++ 编译器（cl.exe）也能找到 CUDA 头文件
# CUDAToolkit_INCLUDE_DIRS 是由 find_package(CUDAToolkit) 提供的
target_include_directories(${TARGET_NAME} PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
)


# 编译选项（兼容 Windows 编译器）
target_compile_options(${TARGET_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/source-charset:utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/execution-charset:utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
)

# CUDA 相关设置
set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_ARCHITECTURES 89
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_FAST_MATH ON
)

# 可执行文件输出到 ./build 目录
set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build
)