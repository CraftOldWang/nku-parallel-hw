#include "../md5.h"
#include "../md5_avx.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <immintrin.h>  // AVX 指令集

using namespace std;
using namespace chrono;

// 辅助函数：打印32位整数的十六进制表示
void print_hex(const char* label, bit32 value) {
    cout << label << std::setw(8) << std::setfill('0') << hex << value;
}

// 从 AVX 向量中提取特定索引的值
void extract_avx_values(bit32* values, __m256i* avx_result, int lane_index) {
    // 为每个向量创建临时数组
    alignas(32) bit32 temp0[8], temp1[8], temp2[8], temp3[8];
    
    // 存储完整向量到内存
    _mm256_store_si256((__m256i*)temp0, avx_result[0]);
    _mm256_store_si256((__m256i*)temp1, avx_result[1]);
    _mm256_store_si256((__m256i*)temp2, avx_result[2]);
    _mm256_store_si256((__m256i*)temp3, avx_result[3]);
    
    // 提取特定 lane 的值
    values[0] = temp0[lane_index];
    values[1] = temp1[lane_index];
    values[2] = temp2[lane_index];
    values[3] = temp3[lane_index];
}

// 比较 AVX 和非 AVX 版本的 MD5 结果
bool compare_results(bit32* md5_result, __m256i* avx_result, const int index) {
    bit32 avx_values[4];
    extract_avx_values(avx_values, avx_result, index);
    
    return (md5_result[0] == avx_values[0] &&
            md5_result[1] == avx_values[1] &&
            md5_result[2] == avx_values[2] &&
            md5_result[3] == avx_values[3]);
}

void correct_test_avx() {
    cout << "=== MD5 AVX 正确性测试 ===" << endl << endl;
    
    // 测试用字符串 - 使用不同长度测试并准备8个 (AVX2可以并行处理8个字符串)
    string test_strings[8] = {
        "test string 1",                       // 短字符串
        "This is a medium length test string", // 中等长度
        "bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjk", // 中长字符串
        "bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefa", // 较长字符串
        "bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva", // 长字符串
        "bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabcdef", // 长字符串+
        "bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabcdefghijk", // 长字符串++
        "bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva" // 非常长的字符串
    };
    
    // 分别测试每个字符串
    for (int i = 0; i < 8; i++) {
        cout << "测试字符串 " << i+1 << " (长度: " << test_strings[i].length() << ")" << endl;
        
        // 使用标准 MD5
        bit32 md5_result[4];
        MD5Hash(test_strings[i], md5_result);
        
        cout << "标准 MD5: ";
        for (int j = 0; j < 4; j++) {
            print_hex("", md5_result[j]);
        }
        cout << endl;
        
        // 使用 AVX MD5 (处理8个字符串)
        __m256i avx_result[4];
        MD5Hash_AVX(test_strings, avx_result);
        
        cout << "AVX MD5[" << i << "]: ";
        bit32 avx_values[4];
        extract_avx_values(avx_values, avx_result, i);
        
        for (int j = 0; j < 4; j++) {
            print_hex("", avx_values[j]);
        }
        cout << endl;
        
        // 比较结果
        bool match = compare_results(md5_result, avx_result, i);
        cout << "结果" << (match ? "匹配" : "不匹配") << "!" << endl << endl;
    }
    
    // 测试并行性能提升
    cout << "=== 性能测试 ===" << endl;
    
    const int N_TESTS = 1000;
    
    // 测试标准 MD5
    auto start = system_clock::now();
    for (int i = 0; i < N_TESTS; i++) {
        bit32 md5_result[4];
        for (int j = 0; j < 8; j++) {  // 处理8个字符串以便公平比较
            MD5Hash(test_strings[j], md5_result);
        }
    }
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    double standard_time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    
    cout << "标准 MD5 处理 " << N_TESTS * 8 << " 个字符串用时: " 
         << standard_time << " 秒" << endl;
    
    // 测试 AVX MD5
    start = system_clock::now();
    for (int i = 0; i < N_TESTS; i++) {
        __m256i avx_result[4];
        MD5Hash_AVX(test_strings, avx_result);
    }
    end = system_clock::now();
    duration = duration_cast<microseconds>(end - start);
    double avx_time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    
    cout << "AVX MD5 处理 " << N_TESTS * 8 << " 个字符串用时: " 
         << avx_time << " 秒" << endl;
    
    cout << "加速比: " << (standard_time / avx_time) << "x" << endl;
    
}