cmake_minimum_required(VERSION 3.30)
project(correct_test CXX)
project(main_pool CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(Threads REQUIRED) # 查找系统上的线程库


# 定义所有需要编译的源文件
set(SOURCE_FILES
        guessing_pthread_pool.cpp
        train.cpp
        correctness_guess_pool.cpp
        md5.cpp
)
# 添加可执行文件。'main_pool' 将是你生成的可执行文件的名称。
# ${SOURCE_FILES} 会把上面列表中的所有源文件都包含进来。
add_executable(correct_test ${SOURCE_FILES})

# 链接 pthreads 库。
# 这是现代 CMake 推荐的、跨平台的方式来链接 pthreads。
target_link_libraries(correct_test PRIVATE Threads::Threads) # 将线程库链接到你的可执行文件


# 设置源文件
set(MAIN_POOL_SRCS
        train.cpp
        md5.cpp
        guessing_pthread_pool.cpp
        main_pool.cpp
)

# 添加 main_pool 可执行文件
add_executable(main_pool ${MAIN_POOL_SRCS})
# 链接 pthread 库
target_link_libraries(main_pool PRIVATE Threads::Threads)



add_executable(main_normal
    main.cpp
    train.cpp
    guessing.cpp
    md5.cpp
)


target_link_libraries(main_normal PRIVATE Threads::Threads)
