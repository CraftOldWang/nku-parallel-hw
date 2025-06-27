# CUDA GPU 并行化密码猜测程序

本项目实现了基于 CUDA GPU 的并行化密码猜测算法，对原始`guessing.cpp`中的两个关键 for 循环进行了 GPU 并行化优化。

## 文件说明

### 新增文件

-   `guessing_cuda.cu` - CUDA GPU 并行化实现的核心文件
-   `guessing_cuda.h` - CUDA 版本的头文件声明
-   `main_cuda.cpp` - 使用 CUDA 版本的主程序（基于 main.cpp 结构）
-   `CMakeLists_cuda.txt` - CUDA 版本的 CMake 构建配置
-   `build_cuda.sh` - Linux/Unix 构建脚本
-   `build_cuda.bat` - Windows 构建脚本
-   `README_CUDA.md` - 本说明文件

### 原始文件

-   `guessing.cpp` - 原始的串行版本实现
-   `main.cpp` - 原始的主程序
-   `PCFG.h` - 数据结构和类定义（已添加 CUDA 方法声明）
-   `md5.cpp/md5.h` - MD5 哈希相关实现

## 主程序特点

`main_cuda.cpp` 基于 `main.cpp` 的完整结构实现，包含：

-   完整的实验配置和批处理逻辑
-   CUDA 设备检测和初始化
-   与原版相同的性能测试框架
-   使用 CUDA 并行化的猜测生成（`PopNext_CUDA()`）
-   标准 MD5 哈希计算（可扩展为 CUDA MD5）

## 并行化实现详情

### 并行化的两个关键循环

1. **单段情况下的循环** (guessing.cpp:217-223)
    ```cpp
    for (int i = 0; i < pt.max_indices[0]; i += 1) {
        string guess = a->ordered_values[i];
        guesses.emplace_back(guess);
        total_guesses += 1;
    }
    ```
2. **多段情况下的循环** (guessing.cpp:268-278)
    ```cpp
    for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1) {
        string temp = guess + a->ordered_values[i];
        guesses.emplace_back(temp);
        total_guesses += 1;
    }
    ```

### CUDA 并行化策略

-   **线程映射**: 每个 CUDA 线程处理一个密码猜测的生成
-   **内存管理**: 使用 GPU 全局内存存储字符串数据和结果
-   **字符串处理**: 在 GPU 上直接进行字符串拼接操作
-   **批量传输**: 一次性传输所有结果回 CPU，减少内存传输开销

### CUDA Kernel 设计

1. `generateGuessesKernel_SingleSegment`: 处理单段情况
2. `generateGuessesKernel_MultipleSegments`: 处理多段情况，包含前缀拼接

## 编译和运行

### 系统要求

-   NVIDIA GPU (计算能力 6.0+)
-   CUDA Toolkit 11.0+
-   CMake 3.18+
-   C++11 兼容编译器

### 方法 1: 使用 CMake (推荐)

```bash
# 创建构建目录
mkdir build
cd build

# 配置项目
cmake .. -f ../CMakeLists_cuda.txt -DCMAKE_BUILD_TYPE=Release

# 编译
cmake --build .

# 运行
./main_cuda
```

### 方法 2: 直接使用 nvcc

```bash
# 编译CUDA版本
nvcc -o main_cuda main_cuda.cpp guessing_cuda.cu md5.cpp -I. -std=c++11

# 运行
./main_cuda
```

### Windows 环境

```cmd
# 使用Visual Studio的nvcc
nvcc -o main_cuda.exe main_cuda.cpp guessing_cuda.cu md5.cpp -I. -std=c++11

# 运行
main_cuda.exe
```

## 性能优化建议

1. **GPU 选择**: 使用较新的 GPU 架构可获得更好性能
2. **块大小调整**: 可以尝试调整 CUDA kernel 中的 blockSize 参数
3. **内存对齐**: 确保数据在 GPU 内存中正确对齐
4. **流水线**: 对于大规模数据，可以考虑使用 CUDA 流进行流水线处理

## 使用示例

```cpp
#include "guessing_cuda.h"

int main() {
    PriorityQueue_CUDA pq_cuda;

    // 加载模型
    pq_cuda.m.load("model.txt");

    // 初始化
    pq_cuda.init();

    // 使用CUDA并行生成密码
    while (!pq_cuda.priority.empty()) {
        pq_cuda.PopNext_CUDA();  // 使用GPU并行版本
    }

    cout << "生成了 " << pq_cuda.total_guesses << " 个密码猜测" << endl;
    return 0;
}
```

## 注意事项

1. **模型文件**: 确保有正确的模型文件 (`model.txt`)
2. **GPU 内存**: 大规模数据可能需要较大的 GPU 内存
3. **错误处理**: 程序包含基本的 CUDA 错误检查
4. **兼容性**: 代码兼容 CUDA 11.0+和现代 GPU 架构

## 性能对比

相比原始串行版本，CUDA 并行化版本在以下方面有显著提升：

-   **并行度**: 利用 GPU 的数千个核心同时处理
-   **吞吐量**: 大幅提高密码生成速度
-   **可扩展性**: 可以处理更大规模的密码空间

## 故障排除

### 常见问题

1. **编译错误**: 检查 CUDA Toolkit 安装和环境变量
2. **运行时错误**: 确保 GPU 驱动程序最新
3. **内存不足**: 减少批处理大小或使用更大内存的 GPU

### 调试建议

-   使用 `cuda-gdb` 进行 CUDA 程序调试
-   使用 `nvprof` 或 `nsight` 进行性能分析
-   检查 CUDA 错误码和异常处理

## 扩展功能

未来可以考虑的改进：

1. 多 GPU 支持
2. 动态负载均衡
3. 更高效的字符串处理
4. 与其他并行技术（OpenMP、MPI）结合
