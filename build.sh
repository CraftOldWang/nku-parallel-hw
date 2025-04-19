#!/bin/bash

# main.cpp train.cpp guessing.cpp md5.cpp
# CPP_FILES=(
#     main_simd.cpp
#     md5_simd.cpp
#     guessing.cpp
#     md5.cpp
# )

CPP_FILES=(
    main.cpp
    train.cpp 
    guessing.cpp 
    md5.cpp
)

# 获取脚本的当前目录
SCRIPT_DIR=$(cd $(dirname "$0") && pwd)

# 设置目标文件夹
BUILD_DIR="$SCRIPT_DIR"    # 输出的可执行文件放到脚本所在目录

# 创建目标文件夹（一般已经存在，这里保险起见）
mkdir -p "$BUILD_DIR"

# 拼接完整路径到文件
for i in "${!CPP_FILES[@]}"; do
    CPP_FILES[$i]="$SCRIPT_DIR/${CPP_FILES[$i]}"
done

# 获取优化等级（从命令行参数中取，比如 -O1、-O2、-O3）
OPT_LEVEL="$1"  # 如果你没传参数，这就是空字符串

# 编译指定的 .cpp 文件为一个名为 main 的可执行文件
echo "Compiling with: g++ ${CPP_FILES[*]} -o $BUILD_DIR/main $OPT_LEVEL"
g++ "${CPP_FILES[@]}" -o "$BUILD_DIR/main" $OPT_LEVEL

# 检查编译结果
if [ $? -eq 0 ]; then
    echo "Compiled specified .cpp files into $BUILD_DIR/main successfully."
else
    echo "Failed to compile some or all specified files."
fi

echo "Build complete."
