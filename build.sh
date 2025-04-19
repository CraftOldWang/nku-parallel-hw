#!/bin/bash

# main.cpp train.cpp guessing.cpp md5.cpp
CPP_FILES=(
    "main_simd.cpp"
    "md5_simd.cpp"
    "train.cpp"  
    "guessing.cpp"

)   

# 获取脚本的当前目录
SCRIPT_DIR=$(cd $(dirname "$0") && pwd)

# 设置目标文件夹
BUILD_DIR="$SCRIPT_DIR"    # 输出的可执行文件放到脚本所在目录

# 创建目标文件夹（一般已经存在，这里保险起见）
mkdir -p "$BUILD_DIR"

# 手动指定要编译的 .cpp 文件（你可以在这里手动添加需要的文件）
# 例如：CPP_FILES=("file1.cpp" "file2.cpp" "file3.cpp")


# 拼接完整路径到文件
for i in "${!CPP_FILES[@]}"; do
    CPP_FILES[$i]="$SCRIPT_DIR/${CPP_FILES[$i]}"
done
# 编译指定的 .cpp 文件为一个名为 main 的可执行文件
g++ "${CPP_FILES[@]}" -o "$BUILD_DIR/main"

# 检查编译结果
if [ $? -eq 0 ]; then
    echo "Compiled specified .cpp files into $BUILD_DIR/main successfully."
else
    echo "Failed to compile some or all specified files."
fi

echo "Build complete."
