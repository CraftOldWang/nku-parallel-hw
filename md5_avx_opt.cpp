#include "md5_avx.h"
#include "md5.h"
#include <immintrin.h>
#include <cstring>
#include <iostream>

// 可能有问题， 还没有改。。。
// 外部函数声明
extern Byte *StringProcess(string input, int *n_byte);

// 加载数据块到AVX向量的优化函数
inline void load_block(__m256i* x, Byte** paddedMessages, int i, int* blockCounts) {
    // 使用对齐的临时缓冲区以提高性能
    alignas(32) bit32 values[8]; 
    
    for (int j = 0; j < 16; j++) {
        // 清零临时缓冲区
        memset(values, 0, sizeof(values));
        
        // 只处理未超出块数的字符串
        for (int k = 0; k < 8; k++) {
            if (i < blockCounts[k]) {
                // 从正确的偏移量读取数据并组装32位值
                int offset = 4 * j + i * 64;
                values[k] = (paddedMessages[k][offset]) |
                          (paddedMessages[k][offset + 1] << 8) |
                          (paddedMessages[k][offset + 2] << 16) |
                          (paddedMessages[k][offset + 3] << 24);
            }
        }
        
        // 使用对齐加载提高性能
        x[j] = _mm256_load_si256((__m256i*)values);
    }
}

void MD5Hash_AVX(std::string* input, __m256i* state) {
    // 1. 预处理字符串
    const int BATCH_SIZE = 8;
    alignas(32) Byte* paddedMessages[BATCH_SIZE];
    int messageLength[BATCH_SIZE];
    int blockCounts[BATCH_SIZE];
    
    // 2. 初始化和预处理
    int max_blocks = 0;
    for (int i = 0; i < BATCH_SIZE; i++) {
        paddedMessages[i] = StringProcess(input[i], &messageLength[i]);
        blockCounts[i] = messageLength[i] / 64;
        max_blocks = (blockCounts[i] > max_blocks) ? blockCounts[i] : max_blocks;
    }
    
    // 3. 初始化MD5状态 - 使用常量值初始化所有通道
    state[0] = _mm256_set1_epi32(0x67452301);
    state[1] = _mm256_set1_epi32(0xefcdab89);
    state[2] = _mm256_set1_epi32(0x98badcfe);
    state[3] = _mm256_set1_epi32(0x10325476);
    
    // 4. 主处理循环
    for (int i = 0; i < max_blocks; i++) {
        // 对齐分配消息块数组
        alignas(32) __m256i x[16];
        
        // 加载当前数据块到AVX向量
        load_block(x, paddedMessages, i, blockCounts);
        
        // 保存当前状态
        __m256i a = state[0];
        __m256i b = state[1]; 
        __m256i c = state[2];
        __m256i d = state[3];
        
        // MD5四轮变换 - 未变更，保持算法正确性
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
        
        // 更新状态
        state[0] = _mm256_add_epi32(state[0], a);
        state[1] = _mm256_add_epi32(state[1], b);
        state[2] = _mm256_add_epi32(state[2], c);
        state[3] = _mm256_add_epi32(state[3], d);
    }
    
    // 5. 一次性创建字节顺序转换掩码
    alignas(32) static const uint8_t shuffle_mask[32] = {
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3,
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3
    };
    const __m256i mask = _mm256_load_si256((__m256i*)shuffle_mask);
    
    // 应用字节顺序转换
    for (int i = 0; i < 4; i++) {
        state[i] = _mm256_shuffle_epi8(state[i], mask);
    }
    
    // 6. 清理内存
    for (int i = 0; i < BATCH_SIZE; i++) {
        delete[] paddedMessages[i];
    }
}