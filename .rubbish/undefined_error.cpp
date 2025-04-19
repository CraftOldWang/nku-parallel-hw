#include <arm_neon.h>
#include <stdio.h>

// 检查是否开启NEON 支持？？？
#ifdef __ARM_NEON
#warning "NEON is enabled"
#endif

//  代码检查 提示我 vshlq_n_u32  有问题， 但是可以正常使用。。。。所以其实没事
//  g++ ./.rubbish/undefined_error.cpp  -o main 也会warning， 说明编译器默认开启NEON ?(不太理解)

int main() {
    // 初始化一个向量，4 个 32-bit 无符号整数
    uint32x4_t a = vdupq_n_u32(32);

    // 所有元素左移 2 位，相当于乘以 4
    uint32x4_t result = vshlq_n_u32(a, 2);

    // 输出结果
    uint32_t output[4];
    vst1q_u32(output, result);

    for (int i = 0; i < 4; i++) {
        printf("%u ", output[i]);
    }
    printf("\n");

    return 0;
}