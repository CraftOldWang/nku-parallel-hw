#!/bin/bash

# 宏定义用 -Dxxxxx 传入了
# 一次性跑完所有)
mkdir -p ./cur_result
mkdir -p build
mkdir -p cur_result_all
mkdir -p ./cur_result_all/seetime
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




# avx+ 线程池 + cuda
# 用那个 string_view 的话就一次需要产出 >=100000个 guess
# 100000 1000000 5000000 10000000
CL_PATH="D:\Softwares\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64\cl.exe"
# for bsize in  100000 500000 1000000 5000000 10000000 ; do
for bsize in 1000000 ; do
    # for gptread in 1 4 16 64 ; do
    for gptread in  4 ; do
        # for threadnum in 2 8 32; do
        for threadnum in  16 ; do

            echo "🔧 编译 cuda GPU_BATCH_SIZE=$bsize , guess per thread ${gptread} , thread num ${threadnum}"

            nvcc -ccbin "$CL_PATH" \
                main_cuda_ori.cpp guessing_cuda.cu async_gpu_pipeline.cu guessing.cpp train.cpp md5.cpp md5_avx.cpp \
                -o build/guess_bs${bsize}_gpt${gptread}_trn${threadnum} \
                -std=c++17 \
                -O2 \
                -arch=sm_89 \
                -lcudart \
                -DNDEBUG \
                -DGUESS_PER_THREAD=${gptread} \
                -DGPU_BATCH_SIZE=${bsize} \
                -DUSING_SIMD \
                -DUSING_POOL \
                -DTHREAD_NUM=${threadnum} \
                --use_fast_math \
                -Xcompiler "/source-charset:utf-8  /execution-charset:utf-8 /arch:AVX2 /EHsc"
                # -lpthread \

                # -Xcompiler "-mavx -mavx2 -mfma -pthread" \

            if [ $? -ne 0 ]; then
                echo "❌ 编译失败，跳过 batch size = $bsize , guess per thread ${gptread} , thread num ${threadnum}"
                continue
            fi

            echo "🚀 运行 GPU_BATCH_SIZE=$bsize , guess per thread ${gptread} , thread num ${threadnum}"
            ./build/guess_bs${bsize}_gpt${gptread}_trn${threadnum}  > ./cur_result_all/seetime/result_${bsize}_${gptread}_${threadnum}.txt
            echo "✅ 输出保存到 result_${bsize}_gpt${gptread}_thn${threadnum}.txt"
        done
    done
done




# 测试猜测数量
# for bsize in  100000 500000 1000000 5000000 10000000 ; do
# for bsize in  1000000  ; do
#     # for gptread in 1 4 16 64 ; do
#     for gptread in  16; do
#         # for threadnum in 2 8 32; do
#         for threadnum in  8 ; do
# echo "🔧 编译 cuda GPU_BATCH_SIZE=$bsize , guess per thread ${gptread} , thread num ${threadnum}"

# nvcc -ccbin "$CL_PATH" \
#     correctness_guess.cpp guessing_cuda.cu guessing.cpp train.cpp md5.cpp md5_avx.cpp \
#     -o build/guess_correct_test_bs${bsize}_gpt${gptread}_trn${threadnum} \
#     -std=c++17 \
#     -O2 \
#     -arch=sm_89 \
#     -lcudart \
#     -DNDEBUG \
#     -DGUESS_PER_THREAD=${gptread} \
#     -DGPU_BATCH_SIZE=${bsize} \
#     -DUSING_SIMD \
#     -DUSING_POOL \
#     -DTHREAD_NUM=${threadnum} \
#     --use_fast_math \
#     -Xcompiler "/source-charset:utf-8  /execution-charset:utf-8 /arch:AVX2 /EHsc"
#     # -lpthread \

#     # -Xcompiler "-mavx -mavx2 -mfma -pthread" \

# if [ $? -ne 0 ]; then
#     echo "❌ 编译失败，跳过 batch size = $bsize , guess per thread ${gptread} , thread num ${threadnum}"
#     continue
# fi

# echo "🚀 运行 GPU_BATCH_SIZE=$bsize , guess per thread ${gptread} , thread num ${threadnum}"
# ./build/guess_correct_test_bs${bsize}_gpt${gptread}_trn${threadnum}  > ./cur_result_all/seetime/result_correct_test${bsize}_${gptread}_${threadnum}.txt
# echo "✅ 输出保存到 result_correct_test${bsize}_gpt${gptread}_thn${threadnum}.txt"
# done
# done
# done