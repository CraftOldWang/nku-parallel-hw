cmake_minimum_required(VERSION 3.18)
project(GuessPasswordCUDA LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找CUDA
find_package(CUDA REQUIRED)

# 设置CUDA架构（根据你的GPU调整）
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})

# 源文件
set(SOURCES
    main_cuda.cpp
    guessing_cuda.cu
    train.cpp
    guessing.cpp
    md5.cpp
    # 如果需要其他源文件，请在此添加
)

# 头文件
set(HEADERS
    PCFG.h
    guessing_cuda.h
    md5.h
    # 如果需要其他头文件，请在此添加
)

# 创建可执行文件
add_executable(main_cuda ${SOURCES} ${HEADERS})

# 设置CUDA属性
set_property(TARGET main_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# 链接CUDA库
target_link_libraries(main_cuda ${CUDA_LIBRARIES})

# 编译选项
target_compile_options(main_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
    >
)

# 如果是Debug模式，添加调试信息
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(main_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
        $<$<COMPILE_LANGUAGE:CXX>:-g>
    )
endif()

# 如果是Release模式，添加优化选项
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(main_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
endif()

# 打印一些有用的信息
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Toolkit Root Dir: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA Include Dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")

# 构建说明
message(STATUS "")
message(STATUS "构建说明:")
message(STATUS "1. 创建构建目录: mkdir build && cd build")
message(STATUS "2. 配置项目: cmake .. -DCMAKE_BUILD_TYPE=Release")
message(STATUS "3. 编译项目: make 或 cmake --build .")
message(STATUS "4. 运行程序: ./main_cuda")
message(STATUS "")