#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

const int TIMES = 1000000;

void write_result_to_csv(const vector<int>& array_sizes,
    const vector<int>& curtimes,
    const vector<double>& multichain_unroll16_times)
{
    ofstream csv_file("sum_performance_multichain_unroll16.csv");

    csv_file << "Array_Sizes,Repeat_Times,Multichain_Unroll16_Time\n";

    for (size_t i = 0; i < array_sizes.size(); i++) {
        csv_file << array_sizes[i] << ","
                 << curtimes[i] << ","
                 << multichain_unroll16_times[i] << "\n";
    }

    csv_file.close();
    cout << "Results written to multichain_unroll16_performance.csv" << endl;
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
    // 求和数组大小
    for (int i = 5; i < 20; i++) {
        sizes.push_back(2 << i);
    }
    // vector<int> array_sizes;
    // vector<int> rep_times;
    // vector<double> multi_chain_unrolled16_times;

    for (int size : sizes) {
        // cout << "测试数组长度: " << size << endl;
        int curtime = TIMES / size;
        if (curtime < 100) {
            curtime = 100;
        }
        // cout << "重复次数: " << curtime << endl;

        std::chrono::duration<double> total_time_multi_chain_unrolled_16(0);
        for (int i = 0; i < curtime; i++) {
            int* a = new int[size];
            for (int j = 0; j < size; j++) {
                a[j] = j * j;
            }

            // auto start = std::chrono::high_resolution_clock::now();
            sum_multi_chain_unrolled_16(a, size);
            // auto end = std::chrono::high_resolution_clock::now();
            // total_time_multi_chain_unrolled_16 += end - start;

            delete[] a;
        }

        // array_sizes.push_back(size);
        // rep_times.push_back(curtime);
        // multi_chain_unrolled16_times.push_back(total_time_multi_chain_unrolled_16.count());

        // std::cout << "利用超标量优化, 多路链式循环展开16，执行时间: " << total_time_multi_chain_unrolled_16.count() << " 秒" << std::endl;
        // std::cout << "-----------------------------------------------------" << std::endl;
    }
    // write_result_to_csv(array_sizes, rep_times, multi_chain_unrolled16_times);
}

int main()
{
    system("chcp 65001>nul");
    test_array_size();
    return 0;
}