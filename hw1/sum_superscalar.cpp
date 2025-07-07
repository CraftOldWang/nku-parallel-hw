#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

const int TIMES = 1000000;

void write_result_to_csv(const vector<int>& array_sizes,
    const vector<int>& curtimes,
    const vector<double>& superscalar_times)
{
    ofstream csv_file("sum_performance_superscalar.csv");

    csv_file << "Array_Sizes,Repeat_Times,Superscalar_Time\n";

    for (size_t i = 0; i < array_sizes.size(); i++) {
        csv_file << array_sizes[i] << ","
                 << curtimes[i] << ","
                 << superscalar_times[i] << "\n";
    }

    csv_file.close();
    cout << "Results written to superscalar_performance.csv" << endl;
}

int sum_using_superscalar(int a[], int length)
{
    for (int m = length; m > 1; m /= 2) {
        for (int i = 0; i < m / 2; i++) {
            a[i] = a[i * 2] + a[i * 2 + 1];
        }
    }
    return a[0];
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
    // vector<double> superscalar_times;

    for (int size : sizes) {
        // cout << "测试数组长度: " << size << endl;
        int curtime = TIMES / size;
        if (curtime < 100) {
            curtime = 100;
        }
        // cout << "重复次数: " << curtime << endl;

        std::chrono::duration<double> total_time_superscalar(0);
        for (int i = 0; i < curtime; i++) {
            int* a = new int[size];
            for (int j = 0; j < size; j++) {
                a[j] = j * j;
            }

            // auto start = std::chrono::high_resolution_clock::now();
            sum_using_superscalar(a, size);
            // auto end = std::chrono::high_resolution_clock::now();
            // total_time_superscalar += end - start;

            delete[] a;
        }

        // array_sizes.push_back(size);
        // rep_times.push_back(curtime);
        // superscalar_times.push_back(total_time_superscalar.count());

        // std::cout << "利用超标量优化, 递归，执行时间: " << total_time_superscalar.count() << " 秒" << std::endl;
        // std::cout << "-----------------------------------------------------" << std::endl;
    }
    // write_result_to_csv(array_sizes, rep_times, superscalar_times);
}

int main()
{
    system("chcp 65001>nul");
    test_array_size();
    return 0;
}