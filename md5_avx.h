#pragma once
#include <iostream>
#include <string>
#include <cstring>
#include <immintrin.h> 

using namespace std;

// 每次处理 8个string

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef unsigned int bit32;

// MD5的一系列参数。参数是固定的，其实你不需要看懂这些
#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21

// AVX版本的位操作函数 - 每次处理8个32位整数
#define F_AVX(x, y, z) _mm256_or_si256(_mm256_and_si256((x),(y)), _mm256_and_si256(_mm256_xor_si256(_mm256_set1_epi32(0xFFFFFFFF), (x)),(z)))

#define G_AVX(x, y, z) _mm256_or_si256(_mm256_and_si256((x),(z)), _mm256_and_si256((y),_mm256_xor_si256(_mm256_set1_epi32(0xFFFFFFFF), (z))))

#define H_AVX(x, y, z) _mm256_xor_si256(_mm256_xor_si256((x), (y)), (z))

#define I_AVX(x, y, z) _mm256_xor_si256((y), _mm256_or_si256((x), _mm256_xor_si256(_mm256_set1_epi32(0xFFFFFFFF), (z))))

// 左循环移位 - AVX版本
#define ROTATELEFT_AVX(num, n) \
    (_mm256_or_si256( \
        _mm256_slli_epi32((num), (n)), \
        _mm256_srli_epi32((num), (32-(n))) \
    ))

// AVX版本的MD5轮函数
static inline void FF_AVX(__m256i& a, __m256i b, __m256i c, 
                         __m256i d, __m256i x, int s, bit32 ac) {
    a = _mm256_add_epi32(a, _mm256_add_epi32(_mm256_add_epi32(F_AVX(b, c, d), x), _mm256_set1_epi32(ac)));
    a = ROTATELEFT_AVX(a, s);
    a = _mm256_add_epi32(a, b);
}

static inline void GG_AVX(__m256i& a, __m256i b, __m256i c, 
                         __m256i d, __m256i x, int s, bit32 ac) {
    a = _mm256_add_epi32(a, _mm256_add_epi32(_mm256_add_epi32(G_AVX(b, c, d), x), _mm256_set1_epi32(ac)));
    a = ROTATELEFT_AVX(a, s);
    a = _mm256_add_epi32(a, b);
}

static inline void HH_AVX(__m256i& a, __m256i b, __m256i c, 
                         __m256i d, __m256i x, int s, bit32 ac) {
    a = _mm256_add_epi32(a, _mm256_add_epi32(_mm256_add_epi32(H_AVX(b, c, d), x), _mm256_set1_epi32(ac)));
    a = ROTATELEFT_AVX(a, s);
    a = _mm256_add_epi32(a, b);
}

static inline void II_AVX(__m256i& a, __m256i b, __m256i c, 
                         __m256i d, __m256i x, int s, bit32 ac) {
    a = _mm256_add_epi32(a, _mm256_add_epi32(_mm256_add_epi32(I_AVX(b, c, d), x), _mm256_set1_epi32(ac)));
    a = ROTATELEFT_AVX(a, s);
    a = _mm256_add_epi32(a, b);
}

// AVX版本的MD5哈希函数 - 一次处理8个字符串
void MD5Hash_AVX(string_view *input, __m256i *state);