#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

const int TIMES = 1000000;

void write_result_to_csv(const vector<int>& array_sizes,
    const vector<int>& curtimes,
    const vector<double>& multichain_times)
{
    ofstream csv_file("sum_performance_multichain.csv");

    csv_file << "Array_Sizes,Repeat_Times,Multichain_Time\n";

    for (size_t i = 0; i < array_sizes.size(); i++) {
        csv_file << array_sizes[i] << ","
                 << curtimes[i] << ","
                 << multichain_times[i] << "\n";
    }

    csv_file.close();
    cout << "Results written to sum_performance_multichain.csv" << endl;
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

void test_array_size()
{
    // 求和数组大小
    vector<int> sizes;
    for (int i = 5; i < 20; i++) {
        sizes.push_back(2 << i);
    }
    // vector<int> array_sizes;
    // vector<int> rep_times;
    // vector<double> multichain_times;

    for (int size : sizes) {

        // cout << "测试数组长度: " << size << endl;
        int curtime = TIMES / size;
        if (curtime < 100) {
            curtime = 100;
        }
        // cout << "重复次数: " << curtime << endl;

        // 多路链式
        std::chrono::duration<double> total_time_multi_chain(0);
        for (int i = 0; i < curtime; i++) {
            int* a = new int[size];
            for (int j = 0; j < size; j++) {
                a[j] = j * j;
            }

            // auto start = std::chrono::high_resolution_clock::now();
            sum_multi_chain(a, size);
            // auto end = std::chrono::high_resolution_clock::now();
            // total_time_multi_chain += end - start;

            delete[] a;
        }

        // array_sizes.push_back(size);
        // rep_times.push_back(curtime);
        // multichain_times.push_back(total_time_multi_chain.count());

        // std::cout << "多路链式求和方法, 执行时间: " << total_time_multi_chain.count() << " 秒" << std::endl;
        // std::cout << "平均每次执行时间: " << (total_time_multi_chain.count() / curtime) * 1e9 << " 纳秒" << std::endl;
        // std::cout << "-----------------------------------------------------" << std::endl;
    }
    // write_result_to_csv(array_sizes, rep_times, multichain_times);
}

int main()
{
    system("chcp 65001>nul"); // 改变字符集为 utf-8 防止 终端乱码
    // 测试多个2的幂次大小的数组
    test_array_size();
    return 0;
}