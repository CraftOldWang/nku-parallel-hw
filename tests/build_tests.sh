#!/bin/bash

# 获取脚本的当前目录
SCRIPT_DIR=$(cd $(dirname "$0") && pwd)

# 获取脚本所在目录的父目录
PARENT_DIR=$(dirname "$SCRIPT_DIR")

# 设置源文件夹和目标文件夹
SRC_DIR="$SCRIPT_DIR"      # 源文件夹是 tests/
BUILD_DIR="$PARENT_DIR"    # 可执行文件输出到 tests/ 的父目录

# 创建目标文件夹（一般已经存在，这里保险起见）
mkdir -p "$BUILD_DIR"

# 使用通配符收集所有 .cpp 文件
CPP_FILES=("$SRC_DIR"/*.cpp)

# 编译所有 .cpp 文件为一个名为 main 的可执行文件
g++ "${CPP_FILES[@]}" -o "$BUILD_DIR/main"

# 检查编译结果
if [ $? -eq 0 ]; then
    echo "Compiled all test .cpp files into $BUILD_DIR/main successfully."
else
    echo "Failed to compile some or all test files."
fi

echo "Build complete."
