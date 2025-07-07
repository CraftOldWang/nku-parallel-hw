#include <iostream>
#include <windows.h>

using namespace std;
int LOOP = 1;
const long long int ARRAY_SIZE = 1000000000; // Renamed from SIZE to ARRAY_SIZE
long long int* a;
long long int sum = 0;

void init()
{
    a = new long long int[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++)
        a[i] = i;
    sum = 0;
}
void destroy1()
{
    delete[] a;
    sum = 0;
}
void common()
{
    for (int i = 0; i < ARRAY_SIZE; i++)
        sum += a[i];
}
void optimizationAlgorithm3()
{
    for (int m = ARRAY_SIZE; m > 1; m /= 2)
        for (int i = 0; i < m / 2; i++)
            a[i] = a[i * 2] + a[i * 2 + 1];
    sum = a[0];
}
void optimizationAlgorithm2(int n)
{
    if (n == 1)
        return;
    else {
        for (int i = 0; i < n / 2; i++)
            a[i] += a[n - i - 1];
        n = n / 2;
        optimizationAlgorithm2(n);
    }
}
void optimizationAlgorithm1()
{
    int sum1 = 0, sum2 = 0;
    for (int i = 0; i < ARRAY_SIZE; i += 2) {
        sum1 += a[i];
        sum2 += a[i + 1];
    }
    sum = sum1 + sum2;
}
int main()
{
    init();
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&begin);
    for (int l = 0; l < LOOP; l++) {
        // init();
        common();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&end);
    cout << "common:" << (end - begin) * 1000.0 / freq / LOOP << "ms" << endl;

    long long int begin1, end1, freq1;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq1);
    QueryPerformanceCounter((LARGE_INTEGER*)&begin1);
    for (int l = 0; l < LOOP; l++) {
        // init();
        optimizationAlgorithm1();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&end1);
    cout << "optimizationAlgorithm1:" << (end1 - begin1) * 1000.0 / freq1 / LOOP << "ms" << endl;

    long long int begin2, end2, freq2;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq2);
    QueryPerformanceCounter((LARGE_INTEGER*)&begin2);
    for (int l = 0; l < LOOP; l++) {
        // init();
        optimizationAlgorithm2(ARRAY_SIZE - 1);
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&end2);
    cout << "optimizationAlgorithm2:" << (end2 - begin2) * 1000.0 / freq2 / LOOP << "ms" << endl;

    long long int begin3, end3, freq3;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq3);
    QueryPerformanceCounter((LARGE_INTEGER*)&begin3);
    for (int l = 0; l < LOOP; l++) {
        // init();
        optimizationAlgorithm3();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&end3);
    cout << "optimizationAlgorithm3:" << (end3 - begin3) * 1000.0 / freq3 / LOOP << "ms" << endl;

    return 0;
}
