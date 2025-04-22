#!/bin/bash

# 1. 获取构建模式参数
MODE="$1"
shift
ARGS="$@"

# 2. 判断模式，并调用对应的构建脚本
case "$MODE" in
    test)
        echo "🧪 Building test version..."
        ./tests/build_tests.sh $ARGS -march=native 
        ;;
    normal)
        echo "🛠️ Building normal version..."
        ./build.sh 0 $ARGS -march=native 
        ;;
    simd)
        echo "🚀 Building SIMD version..."
        ./build.sh 1 $ARGS -march=native 
        ;;
    avx)
        echo "⚡ Building AVX version..."
        ./build.sh 2 $ARGS 
        ;;
    *)
        echo "❌ Unknown mode: $MODE"
        echo "Usage: ./sub.sh [test|normal|simd|avx] [build options]"
        exit 1
        ;;
esac

# 3. 打印时间和执行测试
echo "🕒 Test start time: $(date +"%Y-%m-%d %H:%M:%S")"

# 根据情况你可以选择传不同的参数
# 这里统一传 1 1（也可以改成根据 MODE 判断）
./test.sh 1 1

echo "✅ Test end time: $(date +"%Y-%m-%d %H:%M:%S")"
