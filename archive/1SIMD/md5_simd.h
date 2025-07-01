#pragma once
#include <iostream>
#include <string>
#include <cstring>
#include <arm_neon.h>
#include "config.h"


using namespace std;

// 每次处理 4个string?

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef unsigned int bit32;

// #define NOT_USING_STRING_ARR // 使用 单个字符串传参
// #define USING_ALIGNED   // 不过这里也没用到。。。。

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



// 每次处理四个; x,y,z 是 uint32x4_t  (bit32 是unsigned int )
#define F_SIMD(x, y, z) ( vorrq_u32(vandq_u32((x),(y)), vandq_u32(vmvnq_u32(x),(z))))

#define G_SIMD(x, y, z) (vorrq_u32(vandq_u32((x),(z)), vandq_u32((y),vmvnq_u32(z))))

#define H_SIMD(x, y, z) (veorq_u32(veorq_u32((x), (y)), (z)))

#define I_SIMD(x, y, z) (veorq_u32((y), vorrq_u32((x), vmvnq_u32(z))))



// 改写成内联函数; 不太看得懂，gpt说是在 .h 文件里写 static inline
// 返回一个 bit32 x4 向量， 会有返回值优化对吧
// static inline uint32x4_t ROTATELEFT_SIMD(uint32x4_t& num, const int n) {
//     return vorrq_u32( 
//       vshlq_n_u32(num, n), 
//       vshrq_n_u32(num, 32 - n)
//     );
// }



// static inline void FF_SIMD(uint32x4_t& a, uint32x4_t b, uint32x4_t c, 
//                           uint32x4_t d, uint32x4_t x,const int s, bit32 ac){
//   a = vaddq_u32(a, vaddq_u32(vaddq_u32(F_SIMD(b, c, d), x), vdupq_n_u32(ac))); 
//   a = ROTATELEFT_SIMD(a, s); 
//   a = vaddq_u32(a, b); 
// }


// static inline void GG_SIMD(uint32x4_t& a, uint32x4_t b, uint32x4_t c, 
//                           uint32x4_t d, uint32x4_t x,const int s, bit32 ac){
//   a = vaddq_u32(a, vaddq_u32(vaddq_u32(G_SIMD(b, c, d), x), vdupq_n_u32(ac))); 
//   a = ROTATELEFT_SIMD(a, s); 
//   a = vaddq_u32(a, b); 
// }

// // static inline void HH_SIMD(uint32x4_t& a, uint32x4_t& b, uint32x4_t& c, 
// //                           uint32x4_t& d, uint32x4_t& x,const int& s, bit32 ac){
// //   a = vaddq_u32(a, vaddq_u32(vaddq_u32(H_SIMD(b, c, d), x), vdupq_n_u32(ac))); 
// //   a = ROTATELEFT_SIMD(a, s); 
// //   a = vaddq_u32(a, b); 
// // }

// static inline void HH_SIMD(uint32x4_t& a, uint32x4_t b, uint32x4_t c, 
//   uint32x4_t d, uint32x4_t x, const int s, bit32 ac) {
// a = vaddq_u32(a, vaddq_u32(vaddq_u32(H_SIMD(b, c, d), x), vdupq_n_u32(ac))); 
// a = ROTATELEFT_SIMD(a, s); 
// a = vaddq_u32(a, b); 
// }

// static inline void II_SIMD(uint32x4_t& a, uint32x4_t b, uint32x4_t c, 
//                           uint32x4_t d, uint32x4_t x,const int s, bit32 ac){
//     a = vaddq_u32(a, vaddq_u32(vaddq_u32(I_SIMD(b, c, d), x), vdupq_n_u32(ac))); 
//     a = ROTATELEFT_SIMD(a, s); 
//     a = vaddq_u32(a, b); 
// }


#define ROTATELEFT_SIMD(num, n) \
    vorrq_u32( \
        vshlq_n_u32((num), (n)), \
        vshrq_n_u32((num), 32 - (n)) \
    )

#define FF_SIMD(a, b, c, d, x, s, ac)  { \
    (a) = vaddq_u32((a), vaddq_u32(vaddq_u32(F_SIMD((b), (c), (d)), (x)), vdupq_n_u32(ac))); \
    (a) = ROTATELEFT_SIMD((a), (s)); \
    (a) = vaddq_u32((a), (b)); \
} 

#define GG_SIMD(a, b, c, d, x, s, ac)  { \
    (a) = vaddq_u32((a), vaddq_u32(vaddq_u32(G_SIMD((b), (c), (d)), (x)), vdupq_n_u32(ac))); \
    (a) = ROTATELEFT_SIMD((a), (s)); \
    (a) = vaddq_u32((a), (b)); \
} 

#define HH_SIMD(a, b, c, d, x, s, ac)  { \
    (a) = vaddq_u32((a), vaddq_u32(vaddq_u32(H_SIMD((b), (c), (d)), (x)), vdupq_n_u32(ac))); \
    (a) = ROTATELEFT_SIMD((a), (s)); \
    (a) = vaddq_u32((a), (b)); \
} 

#define II_SIMD(a, b, c, d, x, s, ac) { \
    (a) = vaddq_u32((a), vaddq_u32(vaddq_u32(I_SIMD((b), (c), (d)), (x)), vdupq_n_u32(ac))); \
    (a) = ROTATELEFT_SIMD((a), (s)); \
    (a) = vaddq_u32((a), (b)); \
}



#ifdef NOT_USING_STRING_ARR
void MD5Hash_SIMD(string &input1,string &input2, string & input3, string& input4,  uint32x4_t *state);
#endif

#ifndef NOT_USING_STRING_ARR
void MD5Hash_SIMD(string *input,  uint32x4_t *state);
#endif
