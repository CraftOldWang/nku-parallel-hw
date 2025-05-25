#!/bin/bash

# 获取脚本的当前目录
SCRIPT_DIR=$(cd $(dirname "$0") && pwd)

# 设置目标文件夹（输出的可执行文件放到脚本所在目录）
BUILD_DIR="$SCRIPT_DIR"
mkdir -p "$BUILD_DIR"

# 根据第一个参数选择源文件版本
VERSION="$1"
shift  # 移除第一个参数，保留后续参数作为编译选项

# 根据版本选择文件
if [[ "$VERSION" == "0" ]]; then
    echo "🔧 编译：普通版本"
    CPP_FILES=(
        main.cpp
        train.cpp 
        guessing.cpp 
        md5.cpp
    )
    OUTPUT_FILE="main"
elif [[ "$VERSION" == "1" ]]; then
    echo "🔧 编译：pthread 动态线程版本？？"
    CPP_FILES=(
        main.cpp
        md5.cpp
        guessing_pthread.cpp
        train.cpp
    )
    EXTRA_FLAGS= "-pthread"
    OUTPUT_FILE="main"
elif [[ "$VERSION" == "2" ]]; then
    echo "🔧 编译：pthread 静态线程？？//注意后面可能需要结合simd , 以及要搞openmp"
    CPP_FILES=(
        # main_win.cpp
        # train.cpp 
        # guessing.cpp 
        # md5.cpp
    )
    OUTPUT_FILE="main"
elif [[ "$VERSION" == "3" ]]; then
    echo "🔧 编译：openmp"
    CPP_FILES=(
        correctness_guess.cpp
        md5.cpp
        guessing_openmp.cpp
        train.cpp
    )
    OUTPUT_FILE="main_omp"
    EXTRA_FLAGS="-fopenmp"
else
    echo "❌ 无效的版本参数。请使用 0（普通）、1（pthread）、2（AVX）"
    exit 1
fi

# 为每个文件添加完整路径
for i in "${!CPP_FILES[@]}"; do
    CPP_FILES[$i]="$SCRIPT_DIR/${CPP_FILES[$i]}"
done

# 所有剩余参数作为用户提供的编译选项
USER_COMPILE_OPTIONS="$@"

# 执行编译
echo "📦 编译命令: g++ ${CPP_FILES[*]} -o $BUILD_DIR/$OUTPUT_FILE $EXTRA_FLAGS $USER_COMPILE_OPTIONS"
g++ "${CPP_FILES[@]}" -o "$BUILD_DIR/$OUTPUT_FILE" $EXTRA_FLAGS $USER_COMPILE_OPTIONS

# 编译结果检查
if [ $? -eq 0 ]; then
    echo "✅ 编译成功：$BUILD_DIR/$OUTPUT_FILE"
else
    echo "❌ 编译失败"
fi

echo "🔚 Build complete."




# 3. 打印时间和执行测试
echo "🕒 Test start time: $(date +"%Y-%m-%d %H:%M:%S")"

# 根据情况你可以选择传不同的参数
# 这里统一传 1 1（也可以改成根据 MODE 判断）
./test.sh 2 1 8



echo "✅ Test end time: $(date +"%Y-%m-%d %H:%M:%S")"


