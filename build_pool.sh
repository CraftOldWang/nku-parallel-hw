#!/bin/bash

# 编译线程池版本的PCFG程序

echo "编译线程池版本的PCFG程序..."

# 编译命令
g++ -O2 main_pool.cpp train.cpp guessing_pthread_pool.cpp md5.cpp -o main_pool -lpthread

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo "编译成功！可执行文件：main_pool"
    echo ""
    echo "使用方法："
    echo "  ./main_pool"
    echo ""
    echo "编译优化级别："
    echo "  无优化: g++ main_pool.cpp train.cpp guessing_pthread_pool.cpp md5.cpp -o main_pool -lpthread"
    echo "  O1优化: g++ -O1 main_pool.cpp train.cpp guessing_pthread_pool.cpp md5.cpp -o main_pool -lpthread"
    echo "  O2优化: g++ -O2 main_pool.cpp train.cpp guessing_pthread_pool.cpp md5.cpp -o main_pool -lpthread"
else
    echo "编译失败！请检查错误信息。"
    exit 1
fi
