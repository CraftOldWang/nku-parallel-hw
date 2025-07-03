#!/bin/bash

# 宏定义用 -Dxxxxx 传入了
# 一次性跑完所有)
mkdir -p ./cur_result
mkdir -p build

# for bsize in 10000; do
#     echo "🔧 编译 普通的-O2"

#     g++ main.cpp guessing_ori.cpp train.cpp md5.cpp \
#         -o build/normal \
#         -std=c++14 \
#         -O2 \

#     if [ $? -ne 0 ]; then
#         echo "❌ 编译失败，跳过 普通的-O2"
#         continue
#     fi

#     echo "🚀 运行 GPU_BATCH_SIZE=$bsize"
#     ./build/normal > ./cur_result/normal_output_testlong.txt
#     echo "✅ 输出保存到 normal_output_testlong.txt"
# done



# #TODO 试试 1 10 100 1000 10000 100000 1000000 10000000
# # 遍历三个不同的 batch size
# # 用那个 string_view 的话就一次需要产出 >=100000个 guess
# for bsize in  100000 1000000 5000000 10000000 ; do
#     echo "🔧 编译 cuda GPU_BATCH_SIZE=$bsize"

#     nvcc main_cuda_ori.cpp guessing_cuda.cu guessing.cpp train.cpp md5.cpp \
#         -o build/guess_bs${bsize} \
#         -std=c++17 \
#         -O2 \
#         -arch=sm_75 \
#         -lcudart \
#         -DNDEBUG \
#         -DGUESS_PER_THREAD=1 \
#         -DGPU_BATCH_SIZE=${bsize} \
#         --use_fast_math \

#     if [ $? -ne 0 ]; then
#         echo "❌ 编译失败，跳过 batch size = $bsize"
#         continue
#     fi

#     echo "🚀 运行 GPU_BATCH_SIZE=$bsize ， reserve 并 emplace_back"
#     ./build/guess_bs${bsize} > ./cur_result/result_${bsize}last_ver.txt
#     echo "✅ 输出保存到 result_${bsize}last_ver.txt"
# done


# avx+ 线程池 + cuda
# 用那个 string_view 的话就一次需要产出 >=100000个 guess
for bsize in  100000 1000000 5000000 10000000 ; do
    echo "🔧 编译 cuda GPU_BATCH_SIZE=$bsize"

    nvcc main_cuda_ori.cpp guessing_cuda.cu guessing.cpp train.cpp md5.cpp md5_avx.cpp \
        -o build/guess_bs${bsize} \
        -std=c++17 \
        -O2 \
        -arch=sm_75 \
        -lcudart \
        -lpthread \
        -DNDEBUG \
        -DGUESS_PER_THREAD=1 \
        -DGPU_BATCH_SIZE=${bsize} \
        -DUSING_SIMD \
        -DUSING_POOL \
        -DTHREAD_NUM=8 \
        --use_fast_math \
        -Xcompiler "-mavx -mavx2 -mfma -pthread"

    if [ $? -ne 0 ]; then
        echo "❌ 编译失败，跳过 batch size = $bsize"
        continue
    fi

    echo "🚀 运行 GPU_BATCH_SIZE=$bsize ， reserve 并 emplace_back"
    ./build/guess_bs${bsize} > ./cur_result/result_${bsize}last_ver.txt
    echo "✅ 输出保存到 result_${bsize}last_ver.txt"
done


# 遍历 guess_per_thread GUESS_PER_THREAD 
# for gsize in 1 2 4 8 16 32; do
#     echo "🔧 编译 cuda GPU_BATCH_SIZE=100_0000 GUESS_PER_THREAD=$gsize"

#     nvcc main_cuda_ori.cpp guessing_cuda.cu guessing.cpp train.cpp md5.cpp \
#         -o build/guess_gs${gsize} \
#         -std=c++14 \
#         -O2 \
#         -arch=sm_75 \
#         -lcudart \
#         -DNDEBUG \
#         -DGUESS_PER_THREAD=${gsize} \
#         -DGPU_BATCH_SIZE=1000000 \
#         --use_fast_math \

#     if [ $? -ne 0 ]; then
#         echo "❌ 编译失败，跳过 guess size = $gsize"
#         continue
#     fi

#     echo "🚀 运行 GUESS_PER_THREAD=$gsize GPU_BATCH_SIZE=100_0000"
#     ./build/guess_gs${gsize} > ./cur_result/result_${gsize}guess_size.txt
#     echo "✅ 输出保存到 result_${gsize}guess_size.txt"
# done

# # 遍历三个不同的 batch size
# # 使用更抽象、结构化的main函数
# for bsize in 10000 100000 1000000; do
#     echo "🔧 编译 cuda with new main GPU_BATCH_SIZE=$bsize"

#     nvcc main_cuda.cpp guessing_cuda.cu guessing.cpp train.cpp md5.cpp \
#         -o build/guess_bs${bsize} \
#         -std=c++14 \
#         -O2 \
#         -arch=sm_75 \
#         -lcudart \
#         -DNDEBUG \
#         -DGPU_BATCH_SIZE=${bsize} \
#         --use_fast_math \

#     if [ $? -ne 0 ]; then
#         echo "❌ 编译失败，跳过 batch size = $bsize"
#         continue
#     fi

#     echo "🚀 运行 GPU_BATCH_SIZE=$bsize"
#     ./build/guess_bs${bsize} > ./cur_result/new_result_${bsize}.txt
#     echo "✅ 输出保存到 new_result_${bsize}.txt"
# done



# # 遍历三个不同的 batch size
# # avx 使用
# for bsize in 10000 100000 1000000; do
#     echo "🔧 编译 avx cuda GPU_BATCH_SIZE=$bsize"

#     nvcc main_cuda_ori.cpp guessing_cuda.cu guessing.cpp train.cpp md5.cpp md5_avx.cpp\
#         -o build/avx_test${bsize} \
#         -std=c++14 \
#         -O2 \
#         -arch=sm_75 \
#         -lcudart \
#         -DNDEBUG \
#         -DGPU_BATCH_SIZE=${bsize} \
#         -DUSING_SIMD \
#         --use_fast_math \
#         -Xcompiler "-mavx -mavx2 -mfma"

#     if [ $? -ne 0 ]; then
#         echo "❌ 编译失败，跳过 batch size = $bsize"
#         continue
#     fi

#     echo "🚀 运行 GPU_BATCH_SIZE=$bsize"
#     ./build/avx_test${bsize} > ./cur_result/avx_result_${bsize}.txt
#     echo "✅ 输出保存到 avx_result_${bsize}.txt"
# done



# 单纯avx
# for bsize in 1000000; do
#     echo "🔧 编译 only avx GPU_BATCH_SIZE=$bsize"


#     g++ main_avx.cpp guessing_ori.cpp train.cpp md5.cpp md5_avx.cpp \
#         -o build/avx_test_only \
#         -std=c++14 \
#         -O2 \
#         -DUSING_SIMD \
#         -mavx -mavx2 -mfma

#     if [ $? -ne 0 ]; then
#         echo "❌ 编译失败，跳过 batch size = $bsize"
#         continue
#     fi

#     echo "🚀 运行 GPU_BATCH_SIZE=$bsize"
#     ./build/avx_test_only > ./cur_result/avx_result_only.txt
#     echo "✅ 输出保存到 avx_result_only.txt"
# done

# # 测一下正确性
# for bsize in 1000000; do
#     echo "🔧 编译 GPU_BATCH_SIZE=$bsize"

#     nvcc correctness_guess.cpp guessing_cuda.cu guessing.cpp train.cpp md5.cpp \
#         -o build/correct_test${bsize} \
#         -std=c++14 \
#         -O2 \
#         -arch=sm_75 \
#         -lcudart \
#         -DNDEBUG \
#         -DGPU_BATCH_SIZE=${bsize} \
#         --use_fast_math

#     if [ $? -ne 0 ]; then
#         echo "❌ 编译失败，跳过 batch size = $bsize"
#         continue
#     fi

#     echo "🚀 运行 GPU_BATCH_SIZE=$bsize"
#     ./build/correct_test${bsize} > ./cur_result/correct_test${bsize}.txt
#     echo "✅ 输出保存到 correct_test${bsize}.txt"
# done
