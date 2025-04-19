#include "../PCFG.h"
#include <chrono>
#include <fstream>
#include "../md5.h"
#include "../md5_simd.h"
#include <iomanip>
#include <random>
#include <assert.h>
#include <arm_neon.h>
#include <stdint.h>

using namespace std;
using namespace chrono;

#define s11 7


typedef uint32x4_t bit32x4;

void print_vector_bit32x4(const bit32x4& res_vec){
    bit32 tmp[4];
    vst1q_u32(tmp, res_vec);
    for (int i = 0; i < 4; i++)
    {
        cout<<"res_vec[" << i << "]: "<< tmp[i]<<endl;
    }
    cout<<endl;
    
}

void assert_vec_eq_uint(const bit32x4& res_vec, const bit32 & res){
    bit32 tmp[4];
    vst1q_u32(tmp, res_vec);
    for(int i=0 ;i<4;i++){
        assert(tmp[i] == res);
    }
}

void test_nine_func(){
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

    // 测试 FGHI
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

    // 测试 ROTATELEFT  FFGGHHII
    {

        // cout <<"a:"<<a<<endl;
        // print_vector_bit32x4(aa);
        
        res = ROTATELEFT(a, 3);
        res_vec = ROTATELEFT_SIMD(aa,3);
        assert_vec_eq_uint(res_vec,res);

        // cout<< a <<endl
        //     << b <<endl
        //     << c <<endl
        //     << d <<endl;

        // print_vector_bit32x4(aa);
        // print_vector_bit32x4(bb);
        // print_vector_bit32x4(cc);
        // print_vector_bit32x4(dd);

        bit32x4 m;
        // FF_SIMD 有问题，而不是
        FF(a,b,c,d,arr[7], s11, 0xd76aa478);
        m = vdupq_n_u32(arr[7]); // 因为一次使用 4个字符串 的 块
        FF_SIMD(aa,bb,cc,dd, m, s11, 0xd76aa478);     
        assert_vec_eq_uint(aa,a); // 只更改了 aa 和 a


        // cout<< a <<endl
        // << b <<endl
        // << c <<endl
        // << d <<endl;
        // print_vector_bit32x4(aa);
        // print_vector_bit32x4(bb);
        // print_vector_bit32x4(cc);
        // print_vector_bit32x4(dd);

        GG(a,b,c,d,arr[7], s11, 0xd76aa478);
        m = vdupq_n_u32(arr[7]); // 因为一次使用 4个字符串 的 块
        GG_SIMD(aa,bb,cc,dd, m, s11, 0xd76aa478);     
        assert_vec_eq_uint(aa,a); // 只更改了 aa 和 a

        HH(a,b,c,d,arr[7], s11, 0xd76aa478);
        m = vdupq_n_u32(arr[7]); // 因为一次使用 4个字符串 的 块
        HH_SIMD(aa,bb,cc,dd, m, s11, 0xd76aa478);     
        assert_vec_eq_uint(aa,a); // 只更改了 aa 和 a

        II(a,b,c,d,arr[7], s11, 0xd76aa478);
        m = vdupq_n_u32(arr[7]); // 因为一次使用 4个字符串 的 块
        II_SIMD(aa,bb,cc,dd, m, s11, 0xd76aa478);     
        assert_vec_eq_uint(aa,a); // 只更改了 aa 和 a


        
        cout<< a <<endl
        << b <<endl
        << c <<endl
        << d <<endl;
        print_vector_bit32x4(aa);
        print_vector_bit32x4(bb);
        print_vector_bit32x4(cc);
        print_vector_bit32x4(dd);

        
    }
    

    // cout <<"test assert ";
    // int kk = 10;
    // assert(kk == 10);
    // assert(kk == 0);


}