#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include "md5_simd.h"
#include <iomanip>
#include <random>
#include <assert.h>
#include <arm_neon.h>
#include <stdint.h>

using namespace std;
using namespace chrono;

#define s11 7

typedef uint32x4_t bit32x4;

int main(){
    bit32 arr[100];

    // 创建随机数引擎
    std::random_device rd;  // 获取硬件随机数种子
    std::mt19937 gen(rd()); // Mersenne Twister 算法（常用的伪随机数生成器）
    
    // 设置随机数分布范围
    std::uniform_int_distribution<unsigned int> dis(0, 0xFFFFFFFF); // unsigned int 的范围

    // 生成随机数
    // unsigned int random_number = dis(gen);

    for(int i=0;i<100;i++){
        arr[i] = dis(gen);
    }


    bit32x4 aa = vdupq_n_u32(arr[0]),
            bb = vdupq_n_u32(arr[1]),
            cc = vdupq_n_u32(arr[2]), 
            dd = vdupq_n_u32(arr[3]);
    bit32 a =  arr[0], b = arr[1], c = arr[2], d = arr[3];


    
    bit32 res;
    bit32x4 res_vec;
    bit32 tmp[4];
    {
        res = F(a,b,c);
        res_vec = F_SIMD(aa,bb,cc);     
        
        vst1q_u32(tmp, res_vec);
        for(int i=0 ;i<4;i++){
            assert(tmp[i] == res);
        }

        res = G(a,b,c);
        res_vec = G_SIMD(aa,bb,cc);
        vst1q_u32(tmp, res_vec);
        for(int i=0 ;i<4;i++){
            assert(tmp[i] == res);
        }

        res = H(a,b,c);
        res_vec = H_SIMD(aa,bb,cc);
        vst1q_u32(tmp, res_vec);
        for(int i=0 ;i<4;i++){
            assert(tmp[i] == res);
        }

        res = I(a,b,c);
        res_vec = I_SIMD(aa,bb,cc);
        vst1q_u32(tmp, res_vec);
        for(int i=0 ;i<4;i++){
            assert(tmp[i] == res);
        }
    }

    {

        ROTATELEFT_SIMD(aa, 3);
        FF(a,b,c,d,arr[7], s11, 0xd76aa478);
        bit32x4 m = vdupq_n_u32(arr[7]);
        FF_SIMD(aa,bb,cc,dd, m, s11, 0xd76aa478);     

        vst1q_u32(tmp, aa);
        for(int i=0 ;i<4;i++){
            assert(tmp[i] == res);
        }

        // res = GG(a,b,c);
        // res_vec = GG_SIMD(aa,bb,cc);
        // vst1q_u32(tmp, res_vec);
        // for(int i=0 ;i<4;i++){
        //     assert(tmp[i] == res);
        // }

        // res = HH(a,b,c);
        // res_vec = HH_SIMD(aa,bb,cc);
        // vst1q_u32(tmp, res_vec);
        // for(int i=0 ;i<4;i++){
        //     assert(tmp[i] == res);
        // }

        // res = II(a,b,c);
        // res_vec = II_SIMD(aa,bb,cc);
        // vst1q_u32(tmp, res_vec);
        // for(int i=0 ;i<4;i++){
        //     assert(tmp[i] == res);
        // }
    }

    return 0;
}