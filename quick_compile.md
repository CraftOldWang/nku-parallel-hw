# 快速编译命令

## Windows (一行命令)

```cmd
nvcc -std=c++11 -O2 -o main_cuda.exe main_cuda.cpp guessing_cuda.cu train.cpp guessing.cpp md5.cpp -I. --expt-relaxed-constexpr --extended-lambda
```

## Linux/Unix (一行命令)

```bash
nvcc -std=c++11 -O2 -o main_cuda main_cuda.cpp guessing_cuda.cu train.cpp guessing.cpp md5.cpp -I. --expt-relaxed-constexpr --extended-lambda
```

## 调试版本

```bash
nvcc -std=c++11 -g -G -o main_cuda_debug main_cuda.cpp guessing_cuda.cu train.cpp guessing.cpp md5.cpp -I. --expt-relaxed-constexpr --extended-lambda
```

## 最高优化版本

```bash
nvcc -std=c++11 -O3 -o main_cuda_fast main_cuda.cpp guessing_cuda.cu train.cpp guessing.cpp md5.cpp -I. --expt-relaxed-constexpr --extended-lambda
```

## 运行

```bash
# Windows
main_cuda.exe

# Linux/Unix
./main_cuda
```
