#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

const int TIMES = 10000;

void print_matrix(double* sum, int N)
{
    for (int i = 0; i < N; i++) {
        cout << sum[i] << " ";
    }
    cout << endl;
}

void write_results_to_csv(const std::vector<int>& matrix_sizes,
    const std::vector<int>& repeat_times,
    // const std::vector<double>& common_times,
    const std::vector<double>& cache_times
    // const std::vector<double>& speedups)
    )
{
    std::ofstream csv_file("matrix_performance_cache.csv");

    // Write header
    csv_file << "Matrix_Size,Repeat_Times,Cache_Optimized_Time\n";

    // Write data
    for (size_t i = 0; i < matrix_sizes.size(); i++) {
        csv_file << matrix_sizes[i] << ","
                 << repeat_times[i] << ","
                //  << common_times[i] << ","
                 << cache_times[i] << ","
                //  << speedups[i] << "\n";
                << "\n";
    }

    csv_file.close();
    std::cout << "Results written to matrix_performance.csv" << std::endl;
}

double* matrix_using_cache(double* vec, double* A, int N)
{
    double* sum = new double[N];
    for (int i = 0; i < N; i++) {
        sum[i] = 0.0;
    }
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            sum[i] += vec[j] * A[j * N + i];
        }
    }
    // print_matrix(sum, N);
    return sum;
}

void test()
{
    const int START_SIZE = 50;
    const int END_SIZE = 5000;
    const int STEP_SIZE = 50;

    // std::vector<int> matrix_sizes;
    // std::vector<int> repeat_times;
    // // std::vector<double> common_times;
    // std::vector<double> cache_times;
    // // std::vector<double> speedups;

    double* ret;

    for (int N = START_SIZE; N <= END_SIZE; N += STEP_SIZE) {
        cout << "测试矩阵大小: " << N << "x" << N << endl;

        // 初始化
        double* A = new double[N * N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = i + j;
            }
        }
        double* vec = new double[N];
        for (int i = 0; i < N; i++) {
            vec[i] = i * i;
        }

        int iterations = std::max(1, TIMES / N);
        // 记录起始时间
        // auto start = std::chrono::high_resolution_clock::now();
        // 重复次数根据矩阵大小调整
        
        // for (int i = 0; i < iterations; i++) {
        //     matrix_common(vec, A, N);
        // }
        // // 记录结束时间
        // auto end = std::chrono::high_resolution_clock::now();
        // // 计算时间差
        // std::chrono::duration<double> elapsed = end - start;

        // // 记录起始时间
        // auto start2 = std::chrono::high_resolution_clock::now();
        // // 使用相同的迭代次数
        for (int i = 0; i < iterations; i++) {
            ret = matrix_using_cache(vec, A, N);
            delete[] ret;
        }
        // // 记录结束时间
        // auto end2 = std::chrono::high_resolution_clock::now();
        // // 计算时间差
        // std::chrono::duration<double> elapsed2 = end2 - start2;


        // // std::cout << "不利用cache优化,执行时间: " << elapsed.count() << " 秒" << std::endl;
        // std::cout << "利用cache优化,执行时间: " << elapsed2.count() << " 秒" << std::endl;
        // // std::cout << "速度提升: " << elapsed.count() / elapsed2.count() << std::endl;
        // std::cout << "-------------------------------------" << std::endl;

        // // 记录结果
        // matrix_sizes.push_back(N);
        // repeat_times.push_back(iterations);
        // // common_times.push_back(elapsed.count());
        // cache_times.push_back(elapsed2.count());
        // // speedups.push_back(elapsed.count() / elapsed2.count());

        // 释放动态分配的内存
        delete[] A;
        delete[] vec;
    }

    // write_results_to_csv(matrix_sizes, repeat_times,  cache_times);
}

int main()
{
    system("chcp 65001>nul");
    test();
    return 0;
}