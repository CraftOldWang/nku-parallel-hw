#!/bin/bash

# CUDA构建脚本 (Linux/Unix版本)
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

# 编译CUDA版本
echo ""
echo "正在编译..."
nvcc -std=c++11 -O2 \
    -o main_cuda \
    main_cuda.cpp \
    guessing_cuda.cu \
    train.cpp \
    guessing.cpp \
    md5.cpp \
    -I. \
    --expt-relaxed-constexpr \
    --extended-lambda

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================="
    echo "编译成功！"
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
    echo "编译失败！"
    echo "请检查错误信息并修复后重试"
    echo "=============================="
    exit 1
fi
