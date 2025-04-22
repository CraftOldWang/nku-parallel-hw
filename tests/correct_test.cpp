#include "../md5.h"
#include "../md5_simd.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <arm_neon.h>


using namespace std;
using namespace chrono;

// 辅助函数：打印32位整数的十六进制表示
void print_hex(const char* label, bit32 value) {
    cout << label << std::setw(8) << std::setfill('0') << hex << value;
}

// 比较 SIMD 和非 SIMD 版本的 MD5 结果
bool compare_results(bit32* md5_result, uint32x4_t* simd_result, const int index) {
    bit32 simd_values[4];
    switch(index) {
        case 0:
            vst1q_lane_u32(&simd_values[0], simd_result[0], 0);
            vst1q_lane_u32(&simd_values[1], simd_result[1], 0);
            vst1q_lane_u32(&simd_values[2], simd_result[2], 0);
            vst1q_lane_u32(&simd_values[3], simd_result[3], 0);
            break;
        case 1:
            vst1q_lane_u32(&simd_values[0], simd_result[0], 1);
            vst1q_lane_u32(&simd_values[1], simd_result[1], 1);
            vst1q_lane_u32(&simd_values[2], simd_result[2], 1);
            vst1q_lane_u32(&simd_values[3], simd_result[3], 1);
            break;
        case 2:
            vst1q_lane_u32(&simd_values[0], simd_result[0], 2);
            vst1q_lane_u32(&simd_values[1], simd_result[1], 2);
            vst1q_lane_u32(&simd_values[2], simd_result[2], 2);
            vst1q_lane_u32(&simd_values[3], simd_result[3], 2);
            break;
        case 3:
            vst1q_lane_u32(&simd_values[0], simd_result[0], 3);
            vst1q_lane_u32(&simd_values[1], simd_result[1], 3);
            vst1q_lane_u32(&simd_values[2], simd_result[2], 3);
            vst1q_lane_u32(&simd_values[3], simd_result[3], 3);
            break;
        default:
            simd_values[0] = 0;
            simd_values[1] = 0;
            simd_values[2] = 0;
            simd_values[3] = 0;
    }

    
    return (md5_result[0] == simd_values[0] &&
            md5_result[1] == simd_values[1] &&
            md5_result[2] == simd_values[2] &&
            md5_result[3] == simd_values[3]);
}

int correct_test() {
    cout << "=== MD5 SIMD 正确性测试 ===" << endl << endl;
    
    // 测试用字符串 - 使用不同长度测试
    alignas(16) string test_strings[4] = {
        "test string 1 jsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjjdfjkwanfdjvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsrergwergweerwwe",                       // 短字符串
        "This is a medium length test stringjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjvjkanbjbjadfajwefajksdfakdnsvjadfasjdvajadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsberwew", // 中等长度
        "bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjvjkanbjbjadfajwefajksdjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsfakdnsvjadfasjdvaberere", // 长字符串
        "bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva" // 非常长的字符串
    };
    
    // 分别测试每个字符串
    for (int i = 0; i < 4; i++) {
        cout << "测试字符串 " << i+1 << " (长度: " << test_strings[i].length() << ")" << endl;
        
        // 使用标准 MD5
        bit32 md5_result[4];
        MD5Hash(test_strings[i], md5_result);
        
        cout << "标准 MD5: ";
        for (int j = 0; j < 4; j++) {
            print_hex("", md5_result[j]);
        }
        cout << endl;
        
        // 使用 SIMD MD5 (需要4个字符串)
        uint32x4_t simd_result[4];
        MD5Hash_SIMD(test_strings[0], test_strings[1], test_strings[2],test_strings[3], simd_result);
        
        cout << "SIMD MD5[" << i << "]: ";
        bit32 simd_values[4];
        // vst1q_lane_u32(&simd_values[0], simd_result[0], i);
        // vst1q_lane_u32(&simd_values[1], simd_result[1], i);
        // vst1q_lane_u32(&simd_values[2], simd_result[2], i);
        // vst1q_lane_u32(&simd_values[3], simd_result[3], i);
        switch(i) {
            case 0:
                vst1q_lane_u32(&simd_values[0], simd_result[0], 0);
                vst1q_lane_u32(&simd_values[1], simd_result[1], 0);
                vst1q_lane_u32(&simd_values[2], simd_result[2], 0);
                vst1q_lane_u32(&simd_values[3], simd_result[3], 0);
                break;
            case 1:
                vst1q_lane_u32(&simd_values[0], simd_result[0], 1);
                vst1q_lane_u32(&simd_values[1], simd_result[1], 1);
                vst1q_lane_u32(&simd_values[2], simd_result[2], 1);
                vst1q_lane_u32(&simd_values[3], simd_result[3], 1);
                break;
            case 2:
                vst1q_lane_u32(&simd_values[0], simd_result[0], 2);
                vst1q_lane_u32(&simd_values[1], simd_result[1], 2);
                vst1q_lane_u32(&simd_values[2], simd_result[2], 2);
                vst1q_lane_u32(&simd_values[3], simd_result[3], 2);
                break;
            case 3:
                vst1q_lane_u32(&simd_values[0], simd_result[0], 3);
                vst1q_lane_u32(&simd_values[1], simd_result[1], 3);
                vst1q_lane_u32(&simd_values[2], simd_result[2], 3);
                vst1q_lane_u32(&simd_values[3], simd_result[3], 3);
                break;
            default:
                simd_values[0] = 0;
                simd_values[1] = 0;
                simd_values[2] = 0;
                simd_values[3] = 0;
        }
        
        for (int j = 0; j < 4; j++) {
            print_hex("", simd_values[j]);
        }
        cout << endl;
        
        // 比较结果
        bool match = compare_results(md5_result, simd_result, i);
        cout << "结果" << (match ? "匹配" : "不匹配") << "!" << endl << endl;
    }
    
    // 测试并行性能提升
    cout << "=== 性能测试 ===" << endl;
    
    const int N_TESTS = 1000;
    
    // 测试标准 MD5
    auto start = system_clock::now();
    for (int i = 0; i < N_TESTS; i++) {
        bit32 md5_result[4];
        for (int j = 0; j < 4; j++) {
            MD5Hash(test_strings[j], md5_result);
        }
    }
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    double standard_time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    
    cout << "标准 MD5 处理 " << N_TESTS * 4 << " 个字符串用时: " 
         << standard_time << " 秒" << endl;
    
    // 测试 SIMD MD5
    start = system_clock::now();
    for (int i = 0; i < N_TESTS; i++) {
        uint32x4_t simd_result[4];
        string &pw1 = test_strings[0];
        string &pw2 = test_strings[1];
        string &pw3 = test_strings[2];
        string &pw4 = test_strings[3];
        MD5Hash_SIMD(pw1,pw2,pw3,pw4, simd_result);
    }
    end = system_clock::now();
    duration = duration_cast<microseconds>(end - start);
    double simd_time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    
    cout << "SIMD MD5 处理 " << N_TESTS * 4 << " 个字符串用时: " 
         << simd_time << " 秒" << endl;
    
    cout << "加速比: " << (standard_time / simd_time) << "x" << endl;
    
    return 0;
}