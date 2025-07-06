cmake_minimum_required(VERSION 3.20)
project(GuessHashCUDA LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_COMPILER "D:/Softwares/CUDA/v12.9/bin/nvcc.exe")




# 查找 CUDA Toolkit
# 这会设置 CUDAToolkit_INCLUDE_DIRS 等变量，用于后续引用 CUDA 头文件
find_package(CUDAToolkit REQUIRED)

# 源码文件列表
set(SOURCES
    ${CMAKE_SOURCE_DIR}/main_cuda_ori.cpp
    ${CMAKE_SOURCE_DIR}/guessing.cpp
    ${CMAKE_SOURCE_DIR}/guessing_cuda.cu
    ${CMAKE_SOURCE_DIR}/train.cpp
    ${CMAKE_SOURCE_DIR}/md5.cpp
    ${CMAKE_SOURCE_DIR}/md5_avx.cpp
)

# 支持外部传入参数，设置默认值（可通过 -D 参数覆盖）
set(BATCH_SIZE 1000000 CACHE STRING "GPU batch size")
set(GUESS_PER_THREAD 4 CACHE STRING "Guesses per thread")
set(THREAD_NUM 8 CACHE STRING "Thread number")

# 目标名称带参数区分
set(TARGET_NAME guess_bs${BATCH_SIZE}_gpt${GUESS_PER_THREAD}_trn${THREAD_NUM})

add_executable(${TARGET_NAME} ${SOURCES})

# 定义宏
target_compile_definitions(${TARGET_NAME} PRIVATE
    GPU_BATCH_SIZE=${BATCH_SIZE}
    GUESS_PER_THREAD=${GUESS_PER_THREAD}
    THREAD_NUM=${THREAD_NUM}
    USING_POOL
    USING_SIMD
    NDEBUG
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Windows 下 C++ 编译选项
target_compile_options(${TARGET_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/source-charset:utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/execution-charset:utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
    $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
)

# CUDA 属性
set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_ARCHITECTURES 89
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_FAST_MATH ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
