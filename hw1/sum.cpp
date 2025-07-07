#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

const int TIMES = 1000000;

void write_result_to_csv(const vector<int>& array_sizes,
    const vector<int>& curtimes,
    const vector<double>& common_times,
    const vector<double>& multichain_times,
    const vector<double>& superscalar_times,
    const vector<double>& multichain_unroll_times,
    const vector<double>& multichain_unroll16_times)
{
    ofstream csv_file("sum_performance_testopt.csv");

    csv_file << "Array_Sizes,Repeat_Times,Common_Time,Multichain_Time,Superscalar_Time,Multichain_Unroll_Time,Multichain_Unroll16_Time\n";

    for (size_t i = 0; i < array_sizes.size(); i++) {
        csv_file << array_sizes[i] << ","
                 << curtimes[i] << ","
                 << common_times[i] << ","
                 << multichain_times[i] << ","
                 << superscalar_times[i] << ","
                 << multichain_unroll_times[i] << ","
                 << multichain_unroll16_times[i] << "\n";
    }

    csv_file.close();
    cout << "Results written to sum_performance.csv" << endl;
}

int sum_common(int a[], int length)
{
    int sum = 0;
    for (int i = 0; i < length; i++) {
        sum += a[i];
    }
    return sum;
}

int sum_multi_chain(int a[], int length)
{
    // 多链路式
    int sum1 = 0;
    int sum2 = 0;
    for (int i = 0; i < length; i += 2) {
        sum1 += a[i];
        sum2 += a[i + 1];
    }

    return sum1 + sum2;
}

int sum_using_superscalar(int a[], int length)
{
    for (int m = length; m > 1; m /= 2) { // log(length)个步骤
        for (int i = 0; i < m / 2; i++) {
            a[i] = a[i * 2] + a[i * 2 + 1]; // 相邻元素相加连续存储到数组最前面
        }
    }
    return a[0];
}

// 多路链式的unroll
int sum_multi_chain_unrolled(int a[], int length)
{
    int sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    for (int i = 0; i < length; i += 4) {
        sum1 += a[i];
        sum2 += a[i + 1];
        sum3 += a[i + 2];
        sum4 += a[i + 3];
    }
    int total = sum1 + sum2 + sum3 + sum4;
    return total;
}

int sum_multi_chain_unrolled_16(int a[], int length)
{
    int sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    int sum5 = 0, sum6 = 0, sum7 = 0, sum8 = 0;
    int sum9 = 0, sum10 = 0, sum11 = 0, sum12 = 0;
    int sum13 = 0, sum14 = 0, sum15 = 0, sum16 = 0;

    for (int i = 0; i < length; i += 16) {
        sum1 += a[i];
        sum2 += a[i + 1];
        sum3 += a[i + 2];
        sum4 += a[i + 3];
        sum5 += a[i + 4];
        sum6 += a[i + 5];
        sum7 += a[i + 6];
        sum8 += a[i + 7];
        sum9 += a[i + 8];
        sum10 += a[i + 9];
        sum11 += a[i + 10];
        sum12 += a[i + 11];
        sum13 += a[i + 12];
        sum14 += a[i + 13];
        sum15 += a[i + 14];
        sum16 += a[i + 15];
    }

    int total = sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + sum8 + sum9 + sum10 + sum11 + sum12 + sum13 + sum14 + sum15 + sum16;
    return total;
}

void test_array_size()
{

    vector<int> sizes;
    for (int i = 5; i < 20; i++) {
        sizes.push_back(2 << i);
    }
    vector<int> array_sizes;
    vector<int> rep_times;

    vector<double> common_times;
    vector<double> multichain_times;
    vector<double> superscalar_times;
    vector<double> multi_chain_unrolled_times;
    vector<double> multi_chain_unrolled16_times;
    // 加速比的几张图，就在python里面画吧。

    for (int size : sizes) {
        
        cout << "测试数组长度: " << size << endl;
        int curtime = TIMES / size;
        if (curtime < 10000) {
            curtime = 10000;
        }
        cout << "重复次数: " << curtime << endl;

        // 普通方式
        std::chrono::duration<double> total_time_common(0);
        for (int i = 0; i < curtime; i++) {
            int* a = new int[size];
            for (int j = 0; j < size; j++) {
                a[j] = j * j;
            }

            auto start = std::chrono::high_resolution_clock::now();
            sum_common(a, size);
            auto end = std::chrono::high_resolution_clock::now();
            total_time_common += end - start;

            delete[] a;
        }

        // 多路链式
        std::chrono::duration<double> total_time_multi_chain(0);
        for (int i = 0; i < curtime; i++) {
            int* a = new int[size];
            for (int j = 0; j < size; j++) {
                a[j] = j * j;
            }

            auto start = std::chrono::high_resolution_clock::now();
            sum_multi_chain(a, size);
            auto end = std::chrono::high_resolution_clock::now();
            total_time_multi_chain += end - start;

            delete[] a;
        }

        // 超标量优化，
        std::chrono::duration<double> total_time_superscalar(0);
        for (int i = 0; i < curtime; i++) {
            int* a = new int[size];
            for (int j = 0; j < size; j++) {
                a[j] = j * j;
            }

            auto start = std::chrono::high_resolution_clock::now();
            sum_using_superscalar(a, size);
            auto end = std::chrono::high_resolution_clock::now();
            total_time_superscalar += end - start;

            delete[] a;
        }

        // 多路链式，循环展开优化
        std::chrono::duration<double> total_time_multi_chain_unrolled(0);
        for (int i = 0; i < curtime; i++) {
            int* a = new int[size];
            for (int j = 0; j < size; j++) {
                a[j] = j * j;
            }

            auto start = std::chrono::high_resolution_clock::now();
            sum_multi_chain_unrolled(a, size);
            auto end = std::chrono::high_resolution_clock::now();
            total_time_multi_chain_unrolled += end - start;

            delete[] a;
        }

        // 多路链式，循环展开优化16
        std::chrono::duration<double> total_time_multi_chain_unrolled_16(0);
        for (int i = 0; i < curtime; i++) {
            int* a = new int[size];
            for (int j = 0; j < size; j++) {
                a[j] = j * j;
            }

            auto start = std::chrono::high_resolution_clock::now();
            sum_multi_chain_unrolled_16(a, size);
            auto end = std::chrono::high_resolution_clock::now();
            total_time_multi_chain_unrolled_16 += end - start;

            delete[] a;
        }

        array_sizes.push_back(size);
        rep_times.push_back(curtime);
        common_times.push_back(total_time_common.count());
        multichain_times.push_back(total_time_multi_chain.count());
        superscalar_times.push_back(total_time_superscalar.count());
        multi_chain_unrolled_times.push_back(total_time_multi_chain_unrolled.count());
        multi_chain_unrolled16_times.push_back(total_time_multi_chain_unrolled_16.count());

        std::cout << "不利用超标量优化, 执行时间: " << total_time_common.count() << " 秒" << std::endl;
        std::cout << "利用超标量优化, 多路链式，执行时间: " << total_time_multi_chain.count() << " 秒" << std::endl;
        std::cout << "利用超标量优化, 递归，执行时间: " << total_time_superscalar.count() << " 秒" << std::endl;
        std::cout << "利用超标量优化, 多路链式循环展开，执行时间: " << total_time_multi_chain_unrolled.count() << " 秒" << std::endl;
        std::cout << "利用超标量优化, 多路链式循环展开16，执行时间: " << total_time_multi_chain_unrolled_16.count() << " 秒" << std::endl;
        std::cout << "-----------------------------------------------------" << std::endl;
    }
    write_result_to_csv(array_sizes, rep_times, common_times, 
        multichain_times, superscalar_times, multi_chain_unrolled_times, multi_chain_unrolled16_times);
}

int main()
{
    system("chcp 65001>nul"); // 改变字符集为 utf-8 防止 终端乱码
    // 测试多个2的幂次大小的数组
    test_array_size();
    return 0;
}
