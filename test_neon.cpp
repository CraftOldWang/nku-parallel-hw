#include <arm_neon.h>
#include <iostream>
#include <stdio.h>
using namespace std;

int main() {
    // 测试 128 位寄存器
    uint32x4_t vec4 = vdupq_n_u32(42);
    uint32_t result4[4];
    vst1q_u32(result4, vec4);
    cout<< "128-bit vector: "<< result4[0]<<" "<< result4[1]<<" "<< result4[2]<<" "<<  result4[3]<<endl;

    // 测试 64 位寄存器
    uint32x2_t vec2 = vdup_n_u32(42);
    uint32_t result2[2];
    vst1_u32(result2, vec2);
    cout<< "64-bit vector: "<< result2[0]<<" "<< result2[1]<<endl;

    return 0;
}