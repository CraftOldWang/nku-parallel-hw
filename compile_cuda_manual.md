# 手动编译 CUDA 程序指南

## 使用 nvcc 直接编译

### Windows (命令提示符或 PowerShell)

```cmd
nvcc -std=c++11 -O2 -o main_cuda.exe ^
    main_cuda.cpp ^
    guessing_cuda.cu ^
    train.cpp ^
    guessing.cpp ^
    md5.cpp ^
    -I. ^
    --expt-relaxed-constexpr ^
    --extended-lambda
```

### Linux/Unix (Bash)

```bash
nvcc -std=c++11 -O2 -o main_cuda \
    main_cuda.cpp \
    guessing_cuda.cu \
    train.cpp \
    guessing.cpp \
    md5.cpp \
    -I. \
    --expt-relaxed-constexpr \
    --extended-lambda
```

## 编译选项说明

-   `-std=c++11`: 使用 C++11 标准
-   `-O2`: 优化级别 2（平衡编译时间和性能）
-   `-o main_cuda[.exe]`: 指定输出文件名
-   `-I.`: 包含当前目录作为头文件搜索路径
-   `--expt-relaxed-constexpr`: 允许在 constexpr 函数中使用扩展功能
-   `--extended-lambda`: 支持扩展的 lambda 表达式特性

## 可选的编译选项

### 调试版本

```bash
# 添加调试信息
nvcc -std=c++11 -g -G -o main_cuda_debug \
    main_cuda.cpp guessing_cuda.cu train.cpp guessing.cpp md5.cpp \
    -I. --expt-relaxed-constexpr --extended-lambda
```

### 高级优化版本

```bash
# 最高优化级别
nvcc -std=c++11 -O3 -o main_cuda_optimized \
    main_cuda.cpp guessing_cuda.cu train.cpp guessing.cpp md5.cpp \
    -I. --expt-relaxed-constexpr --extended-lambda
```

### 指定 GPU 架构

```bash
# 为特定GPU架构编译（例如RTX 30系列）
nvcc -std=c++11 -O2 -arch=sm_86 -o main_cuda \
    main_cuda.cpp guessing_cuda.cu train.cpp guessing.cpp md5.cpp \
    -I. --expt-relaxed-constexpr --extended-lambda
```

常见 GPU 架构代码：

-   sm_60: GTX 10 系列
-   sm_61: GTX 10 系列 (部分型号)
-   sm_70: V100, Titan V
-   sm_75: RTX 20 系列, T4
-   sm_80: A100
-   sm_86: RTX 30 系列
-   sm_89: RTX 40 系列

## 运行程序

### Windows

```cmd
# 运行程序
main_cuda.exe

# 如果需要重定向输出到文件
main_cuda.exe > results.txt
```

### Linux/Unix

```bash
# 运行程序
./main_cuda

# 如果需要重定向输出到文件
./main_cuda > results.txt

# 后台运行
nohup ./main_cuda > results.txt 2>&1 &
```

## 故障排除

### 1. 找不到 nvcc 命令

确保 CUDA Toolkit 已正确安装并添加到 PATH 环境变量：

Windows:

```cmd
# 检查nvcc是否可用
where nvcc

# 如果找不到，添加CUDA bin目录到PATH
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin
```

Linux:

```bash
# 检查nvcc是否可用
which nvcc

# 如果找不到，添加到PATH
export PATH=/usr/local/cuda/bin:$PATH
```

### 2. 编译错误

常见错误和解决方案：

#### 错误: "undefined reference to..."

可能缺少某些源文件，确保包含所有必要的.cpp 文件。

#### 错误: "nvcc fatal : No input files"

检查文件路径是否正确，确保所有源文件都存在。

#### 错误: "ptxas fatal : Unresolved extern function"

可能是 CUDA 代码中有未定义的函数，检查 guessing_cuda.cu 中的函数实现。

### 3. 运行时错误

#### 错误: "no CUDA-capable device is detected"

确保你的系统有 NVIDIA GPU 并安装了正确的驱动程序。

#### 错误: "out of memory"

GPU 内存不足，可以尝试减少批处理大小或优化内存使用。

## 性能优化建议

1. **编译优化**: 使用 `-O3` 而不是 `-O2` 获得更好性能
2. **GPU 架构**: 为你的具体 GPU 架构编译（使用 `-arch=sm_XX`）
3. **批处理大小**: 调整 `main_cuda.cpp` 中的 `batch_size` 参数
4. **内存预分配**: 考虑预分配 GPU 内存以减少内存分配开销

## 示例完整编译流程

```bash
# 1. 检查环境
nvcc --version

# 2. 编译程序
nvcc -std=c++11 -O2 -o main_cuda \
    main_cuda.cpp guessing_cuda.cu train.cpp guessing.cpp md5.cpp \
    -I. --expt-relaxed-constexpr --extended-lambda

# 3. 检查编译结果
ls -la main_cuda  # Linux
dir main_cuda.exe  # Windows

# 4. 运行程序
./main_cuda  # Linux
main_cuda.exe  # Windows
```
