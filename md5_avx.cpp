#include "md5_avx.h"
#include "md5.h"
#include <immintrin.h>
#include <cstring>
#include <string_view>

extern Byte *StringProcess(string_view input, int *n_byte);

void MD5Hash_AVX(string_view *input, __m256i *state)
{
    // 预处理步骤 - 为8个字符串准备填充后的消息
    Byte **paddedMessages = new Byte*[8];
    int *messageLength = new int[8];
    
    // 1. 预处理每个字符串
    for (int i = 0; i < 8; i++) {
        paddedMessages[i] = StringProcess(input[i], &messageLength[i]);
    }
    
    // 2. 计算每个字符串的块数，并找出最大块数
    int blockCounts[8];
    int max_blocks = 0;
    for (int i = 0; i < 8; i++) {
        blockCounts[i] = messageLength[i] / 64; // 每块64字节
        if (blockCounts[i] > max_blocks) {
            max_blocks = blockCounts[i];
        }
    }
    
    // 3. 初始化MD5状态变量
    state[0] = _mm256_set1_epi32(0x67452301);
    state[1] = _mm256_set1_epi32(0xefcdab89);
    state[2] = _mm256_set1_epi32(0x98badcfe);
    state[3] = _mm256_set1_epi32(0x10325476);
    
    // 为已完成的字符串创建临时保存区
    alignas(32) bit32 tmp_state0[8] = {0};
    alignas(32) bit32 tmp_state1[8] = {0};
    alignas(32) bit32 tmp_state2[8] = {0};
    alignas(32) bit32 tmp_state3[8] = {0};
    
    // 4. 主处理循环 - 逐块处理消息
    for (int i = 0; i < max_blocks; i++) {
        __m256i x[16]; // 16个消息块，每个包含8个并行处理的值
        
        // 4.1 加载当前块
        for (int j = 0; j < 16; j++) {
            bit32 values[8] = {0}; // 默认填0
            
            for (int k = 0; k < 8; k++) {
                if (i < blockCounts[k]) {
                    // 对于没有超出块数的字符串，正常加载数据
                    values[k] = (paddedMessages[k][4 * j + i * 64]) |
                               (paddedMessages[k][4 * j + 1 + i * 64] << 8) |
                               (paddedMessages[k][4 * j + 2 + i * 64] << 16) |
                               (paddedMessages[k][4 * j + 3 + i * 64] << 24);
                }
            }
            
            // 将8个值加载到一个向量中
            x[j] = _mm256_loadu_si256((__m256i*)values);
        }
        
        // 4.2 保存初始状态
        __m256i a = state[0], b = state[1], c = state[2], d = state[3];
        
        // 4.3 MD5的4轮运算
        /* Round 1 */
        FF_AVX(a, b, c, d, x[0], s11, 0xd76aa478);
        FF_AVX(d, a, b, c, x[1], s12, 0xe8c7b756);
        FF_AVX(c, d, a, b, x[2], s13, 0x242070db);
        FF_AVX(b, c, d, a, x[3], s14, 0xc1bdceee);
        FF_AVX(a, b, c, d, x[4], s11, 0xf57c0faf);
        FF_AVX(d, a, b, c, x[5], s12, 0x4787c62a);
        FF_AVX(c, d, a, b, x[6], s13, 0xa8304613);
        FF_AVX(b, c, d, a, x[7], s14, 0xfd469501);
        FF_AVX(a, b, c, d, x[8], s11, 0x698098d8);
        FF_AVX(d, a, b, c, x[9], s12, 0x8b44f7af);
        FF_AVX(c, d, a, b, x[10], s13, 0xffff5bb1);
        FF_AVX(b, c, d, a, x[11], s14, 0x895cd7be);
        FF_AVX(a, b, c, d, x[12], s11, 0x6b901122);
        FF_AVX(d, a, b, c, x[13], s12, 0xfd987193);
        FF_AVX(c, d, a, b, x[14], s13, 0xa679438e);
        FF_AVX(b, c, d, a, x[15], s14, 0x49b40821);

        /* Round 2 */
        GG_AVX(a, b, c, d, x[1], s21, 0xf61e2562);
        GG_AVX(d, a, b, c, x[6], s22, 0xc040b340);
        GG_AVX(c, d, a, b, x[11], s23, 0x265e5a51);
        GG_AVX(b, c, d, a, x[0], s24, 0xe9b6c7aa);
        GG_AVX(a, b, c, d, x[5], s21, 0xd62f105d);
        GG_AVX(d, a, b, c, x[10], s22, 0x2441453);
        GG_AVX(c, d, a, b, x[15], s23, 0xd8a1e681);
        GG_AVX(b, c, d, a, x[4], s24, 0xe7d3fbc8);
        GG_AVX(a, b, c, d, x[9], s21, 0x21e1cde6);
        GG_AVX(d, a, b, c, x[14], s22, 0xc33707d6);
        GG_AVX(c, d, a, b, x[3], s23, 0xf4d50d87);
        GG_AVX(b, c, d, a, x[8], s24, 0x455a14ed);
        GG_AVX(a, b, c, d, x[13], s21, 0xa9e3e905);
        GG_AVX(d, a, b, c, x[2], s22, 0xfcefa3f8);
        GG_AVX(c, d, a, b, x[7], s23, 0x676f02d9);
        GG_AVX(b, c, d, a, x[12], s24, 0x8d2a4c8a);

        /* Round 3 */
        HH_AVX(a, b, c, d, x[5], s31, 0xfffa3942);
        HH_AVX(d, a, b, c, x[8], s32, 0x8771f681);
        HH_AVX(c, d, a, b, x[11], s33, 0x6d9d6122);
        HH_AVX(b, c, d, a, x[14], s34, 0xfde5380c);
        HH_AVX(a, b, c, d, x[1], s31, 0xa4beea44);
        HH_AVX(d, a, b, c, x[4], s32, 0x4bdecfa9);
        HH_AVX(c, d, a, b, x[7], s33, 0xf6bb4b60);
        HH_AVX(b, c, d, a, x[10], s34, 0xbebfbc70);
        HH_AVX(a, b, c, d, x[13], s31, 0x289b7ec6);
        HH_AVX(d, a, b, c, x[0], s32, 0xeaa127fa);
        HH_AVX(c, d, a, b, x[3], s33, 0xd4ef3085);
        HH_AVX(b, c, d, a, x[6], s34, 0x4881d05);
        HH_AVX(a, b, c, d, x[9], s31, 0xd9d4d039);
        HH_AVX(d, a, b, c, x[12], s32, 0xe6db99e5);
        HH_AVX(c, d, a, b, x[15], s33, 0x1fa27cf8);
        HH_AVX(b, c, d, a, x[2], s34, 0xc4ac5665);

        /* Round 4 */
        II_AVX(a, b, c, d, x[0], s41, 0xf4292244);
        II_AVX(d, a, b, c, x[7], s42, 0x432aff97);
        II_AVX(c, d, a, b, x[14], s43, 0xab9423a7);
        II_AVX(b, c, d, a, x[5], s44, 0xfc93a039);
        II_AVX(a, b, c, d, x[12], s41, 0x655b59c3);
        II_AVX(d, a, b, c, x[3], s42, 0x8f0ccc92);
        II_AVX(c, d, a, b, x[10], s43, 0xffeff47d);
        II_AVX(b, c, d, a, x[1], s44, 0x85845dd1);
        II_AVX(a, b, c, d, x[8], s41, 0x6fa87e4f);
        II_AVX(d, a, b, c, x[15], s42, 0xfe2ce6e0);
        II_AVX(c, d, a, b, x[6], s43, 0xa3014314);
        II_AVX(b, c, d, a, x[13], s44, 0x4e0811a1);
        II_AVX(a, b, c, d, x[4], s41, 0xf7537e82);
        II_AVX(d, a, b, c, x[11], s42, 0xbd3af235);
        II_AVX(c, d, a, b, x[2], s43, 0x2ad7d2bb);
        II_AVX(b, c, d, a, x[9], s44, 0xeb86d391);
        
        // 4.4 更新状态
        state[0] = _mm256_add_epi32(state[0], a);
        state[1] = _mm256_add_epi32(state[1], b);
        state[2] = _mm256_add_epi32(state[2], c);
        state[3] = _mm256_add_epi32(state[3], d);
        
        // 保存已完成处理的字符串状态
        // 检查每个字符串是否在当前块处理完成
        alignas(32) bit32 state0_values[8], state1_values[8], state2_values[8], state3_values[8];
        _mm256_store_si256((__m256i*)state0_values, state[0]);
        _mm256_store_si256((__m256i*)state1_values, state[1]);
        _mm256_store_si256((__m256i*)state2_values, state[2]);
        _mm256_store_si256((__m256i*)state3_values, state[3]);
        
        // 检查每个字符串是否刚好在此块处理完成
        for (int k = 0; k < 8; k++) {
            if (i == blockCounts[k] - 1) {
                // 该字符串处理完成，保存最终状态
                tmp_state0[k] = state0_values[k];
                tmp_state1[k] = state1_values[k];
                tmp_state2[k] = state2_values[k];
                tmp_state3[k] = state3_values[k];
            }
        }
    }
    
    // 将最终结果从临时保存区加载回状态向量
    state[0] = _mm256_loadu_si256((__m256i*)tmp_state0);
    state[1] = _mm256_loadu_si256((__m256i*)tmp_state1);
    state[2] = _mm256_loadu_si256((__m256i*)tmp_state2);
    state[3] = _mm256_loadu_si256((__m256i*)tmp_state3);
    
    // 5. 字节序转换 (little-endian to big-endian)
    // 使用AVX2的字节重排指令
    __m256i mask = _mm256_set_epi8(
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3,
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3
    );
    for (int i = 0; i < 4; i++) {
        state[i] = _mm256_shuffle_epi8(state[i], mask);
    }
    
    // 6. 清理内存
    for (int i = 0; i < 8; i++) {
        delete[] paddedMessages[i];
    }
    delete[] paddedMessages;
    delete[] messageLength;
}