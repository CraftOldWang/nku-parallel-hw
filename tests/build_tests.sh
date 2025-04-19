#!/bin/bash

# 手动指定父目录中的 .cpp 文件（只写文件名，不含路径）
PARENT_CPP_FILES=( # 你只需要改这里
    md5_simd.cpp
    md5.cpp
    # md5_part_no_macro.cpp
)  

# 获取脚本的当前目录
SCRIPT_DIR=$(cd $(dirname "$0") && pwd)

# 获取脚本所在目录的父目录
PARENT_DIR=$(dirname "$SCRIPT_DIR")

# 设置源文件夹和目标文件夹
SRC_DIR="$SCRIPT_DIR"
BUILD_DIR="$PARENT_DIR"

# 创建目标文件夹（一般已经存在，这里保险起见）
mkdir -p "$BUILD_DIR"

# 可选的优化参数，比如 -O2, -O3，默认不加优化
OPT_FLAG=""
if [[ "$1" =~ ^-O[0-3]$ ]]; then
    OPT_FLAG="$1"
    echo "使用编译优化等级：$OPT_FLAG"
else
    echo "未启用优化编译参数。如需优化，请使用 -O1 / -O2 / -O3 参数。"
fi



# 给这些文件名加上父目录路径
for i in "${!PARENT_CPP_FILES[@]}"; do
    PARENT_CPP_FILES[$i]="$PARENT_DIR/${PARENT_CPP_FILES[$i]}"
done

# 收集 tests/ 下所有 .cpp 文件
CPP_FILES=("$SRC_DIR"/*.cpp)

# 合并全部 cpp 文件
CPP_FILES+=("${PARENT_CPP_FILES[@]}")

# 编译所有 .cpp 文件为一个名为 main 的可执行文件
g++ $OPT_FLAG "${CPP_FILES[@]}" -o "$BUILD_DIR/main"

# 检查编译结果
if [ $? -eq 0 ]; then
    echo "✅ Compiled all test .cpp files into $BUILD_DIR/main successfully."
else
    echo "❌ Failed to compile some or all test files."
fi

echo "Build complete."
