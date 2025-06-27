#!/bin/bash

# CUDA构建脚本 (Linux/Unix版本) - 兼容性修复版
echo "构建CUDA版本的密码猜测程序..."

# 检查CUDA是否可用
if ! command -v nvcc &> /dev/null; then
    echo "错误: NVCC编译器未找到，请确保CUDA Toolkit已安装"
    echo "请检查以下路径是否在PATH环境变量中："
    echo "/usr/local/cuda/bin"
    echo "或者运行: export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi

# 显示CUDA版本信息
echo "CUDA编译器信息:"
nvcc --version

# 检测系统和编译器信息
echo ""
echo "系统信息:"
gcc --version | head -1
echo "检测到的C++标准支持:"

# 尝试多种编译配置
echo ""
echo "正在编译... (尝试C++14标准)"

# 首先尝试C++14标准
nvcc -std=c++14 -O2 \
    -o main_cuda \
    main_cuda.cpp \
    guessing_cuda.cu \
    train.cpp \
    guessing.cpp \
    md5.cpp \
    -I. \
    --expt-relaxed-constexpr \
    --extended-lambda \
    -Xcompiler -fPIC \
    -Xcompiler -Wall \
    -Xcompiler -Wno-unused-result

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================="
    echo "编译成功！ (C++14)"
    echo "可执行文件: main_cuda"
    echo "运行命令: ./main_cuda"
    echo "=============================="
    echo ""
    echo "是否现在运行程序？ (y/n)"
    read -r choice
    if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
        echo "正在运行程序..."
        ./main_cuda
    fi
    exit 0
fi

echo "C++14编译失败，尝试C++11标准..."

# 如果C++14失败，尝试C++11
nvcc -std=c++11 -O2 \
    -o main_cuda \
    main_cuda.cpp \
    guessing_cuda.cu \
    train.cpp \
    guessing.cpp \
    md5.cpp \
    -I. \
    --expt-relaxed-constexpr \
    --extended-lambda \
    -Xcompiler -fPIC \
    -Xcompiler -Wall \
    -Xcompiler -Wno-unused-result

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================="
    echo "编译成功！ (C++11)"
    echo "可执行文件: main_cuda"
    echo "运行命令: ./main_cuda"
    echo "=============================="
    echo ""
    echo "是否现在运行程序？ (y/n)"
    read -r choice
    if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
        echo "正在运行程序..."
        ./main_cuda
    fi
    exit 0
fi

echo "标准编译失败，尝试基础编译..."

# 最后尝试最基础的编译选项
nvcc -O2 \
    -o main_cuda \
    main_cuda.cpp \
    guessing_cuda.cu \
    train.cpp \
    guessing.cpp \
    md5.cpp \
    -I. \
    -Xcompiler -fPIC

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================="
    echo "编译成功！ (基础模式)"
    echo "可执行文件: main_cuda"
    echo "运行命令: ./main_cuda"
    echo "=============================="
    echo ""
    echo "是否现在运行程序？ (y/n)"
    read -r choice
    if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
        echo "正在运行程序..."
        ./main_cuda
    fi
else
    echo ""
    echo "=============================="
    echo "所有编译尝试都失败了！"
    echo "可能的原因："
    echo "1. CUDA版本与系统C++库不兼容"
    echo "2. 缺少某些源文件"
    echo "3. 系统环境配置问题"
    echo ""
    echo "建议："
    echo "1. 检查CUDA版本: nvcc --version"
    echo "2. 检查GCC版本: gcc --version"
    echo "3. 尝试更新CUDA或使用旧版本"
    echo "=============================="
    exit 1
fi
