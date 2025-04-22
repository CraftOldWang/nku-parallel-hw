#!/bin/bash

# simd
# # 手动指定当前文件夹中的 .cpp 文件（只写文件名，不含路径）
# CURRENT_CPP_FILES=(  # 你只需要改这里
#     all_tests.cpp
#     correct_test.cpp
#     test_nine_func.cpp
#     # vrev32q_u8_test.cpp
# )

# # 手动指定父文件夹中的 .cpp 文件（只写文件名，不含路径）
# PARENT_CPP_FILES=(  # 你只需要改这里
#     md5_simd.cpp
#     md5.cpp
#     # md5_part_no_macro.cpp
# )

# avx
CURRENT_CPP_FILES=(  # 你只需要改这里
    all_tests.cpp
    correct_test_avx.cpp
)
PARENT_CPP_FILES=(  # 你只需要改这里

    md5_avx.cpp
    md5.cpp
    # md5_part_no_macro.cpp
)

# 获取脚本的当前目录
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# 获取脚本所在目录的父目录
PARENT_DIR=$(dirname "$SCRIPT_DIR")

# 设置源文件夹和目标文件夹
SRC_DIR="$SCRIPT_DIR"
BUILD_DIR="$PARENT_DIR"

# 创建目标文件夹（一般已经存在，这里保险起见）
mkdir -p "$BUILD_DIR"

# 将所有传入的参数直接作为编译参数
COMPILER_FLAGS="$@"

# 给这些文件名加上父目录路径
for i in "${!CURRENT_CPP_FILES[@]}"; do
    CURRENT_CPP_FILES[$i]="$SRC_DIR/${CURRENT_CPP_FILES[$i]}"
done

for i in "${!PARENT_CPP_FILES[@]}"; do
    PARENT_CPP_FILES[$i]="$PARENT_DIR/${PARENT_CPP_FILES[$i]}"
done

# 合并当前文件夹和父文件夹中的 .cpp 文件
CPP_FILES=("${CURRENT_CPP_FILES[@]}" "${PARENT_CPP_FILES[@]}")

# 编译所有 .cpp 文件为一个名为 main 的可执行文件
echo "Compiling with: g++ ${CPP_FILES[@]} -o $BUILD_DIR/main $COMPILER_FLAGS"
g++ "${CPP_FILES[@]}" -o "$BUILD_DIR/main" $COMPILER_FLAGS

# 检查编译结果
if [ $? -eq 0 ]; then
    echo "✅ Compiled all specified .cpp files into $BUILD_DIR/main successfully."
else
    echo "❌ Failed to compile some or all specified files."
fi


read -p "按回车键继续..." dummy


echo "Build complete."
