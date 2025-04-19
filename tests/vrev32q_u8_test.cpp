// #include <iostream>
// #include <iomanip>
// #include <arm_neon.h>

//NOTE 这个只是单独测试某个 arm_neon 的函数的效果， 不在all_tests 中

// // 辅助函数：打印 32 位整数的十六进制表示
// void print_hex(const char* label, uint32_t value) {
//     std::cout << label << "0x" << std::hex << std::setw(8) 
//               << std::setfill('0') << value << std::dec << std::endl;
// }

// // 辅助函数：打印 128 位 NEON 向量的内容（四个 32 位整数）
// void print_vector(const char* label, uint32x4_t vec) {
//     uint32_t values[4];
//     vst1q_u32(values, vec);
    
//     std::cout << label << std::endl;
//     for (int i = 0; i < 4; i++) {
//         std::cout << "  [" << i << "] = 0x" << std::hex << std::setw(8) 
//                   << std::setfill('0') << values[i] << std::dec << std::endl;
//     }
// }

// // 传统方法：手动字节反转
// uint32_t reverse_bytes_manual(uint32_t value) {
//     return ((value & 0xff) << 24) |      // 将最低字节移到最高位
//            ((value & 0xff00) << 8) |     // 将次低字节左移
//            ((value & 0xff0000) >> 8) |   // 将次高字节右移
//            ((value & 0xff000000) >> 24); // 将最高字节移到最低位
// }

// int main() {
//     // 测试数据：四个 32 位整数
//     uint32_t test_values[4] = {
//         0x12345678,  // 测试值 1
//         0xAABBCCDD,  // 测试值 2
//         0x01020304,  // 测试值 3
//         0xFFEEDDCC   // 测试值 4
//     };
    
//     std::cout << "=== 字节反转测试 ===" << std::endl << std::endl;
    
//     // 1. 使用传统方法反转字节
//     std::cout << "使用传统方法反转字节：" << std::endl;
//     for (int i = 0; i < 4; i++) {
//         uint32_t original = test_values[i];
//         uint32_t reversed = reverse_bytes_manual(original);
        
//         std::cout << "值 " << i << ":" << std::endl;
//         print_hex("  原始值: ", original);
//         print_hex("  反转后: ", reversed);
//         std::cout << std::endl;
//     }
    
//     // 2. 使用 NEON 指令反转字节
//     std::cout << "使用 NEON 指令反转字节：" << std::endl;
    
//     // 加载测试数据到 NEON 向量
//     uint32x4_t vec = vld1q_u32(test_values);
//     print_vector("原始向量:", vec);
    
//     // 将 uint32x4_t 重新解释为 uint8x16_t（字节视图）
//     uint8x16_t bytes = vreinterpretq_u8_u32(vec);
    
//     // 使用 vrev32q_u8 反转每个 32 位组内的字节
//     uint8x16_t reversed_bytes = vrev32q_u8(bytes);
    
//     // 将结果转回 uint32x4_t
//     uint32x4_t reversed_vec = vreinterpretq_u32_u8(reversed_bytes);
//     print_vector("反转后向量:", reversed_vec);
    
//     // 验证 NEON 结果与传统方法一致
//     std::cout << std::endl << "验证 NEON 结果与传统方法一致：" << std::endl;
//     uint32_t neon_results[4];
//     vst1q_u32(neon_results, reversed_vec);
    
//     bool all_match = true;
//     for (int i = 0; i < 4; i++) {
//         uint32_t manual_result = reverse_bytes_manual(test_values[i]);
//         bool match = (neon_results[i] == manual_result);
        
//         std::cout << "值 " << i << ": " 
//                   << (match ? "匹配" : "不匹配") << std::endl;
        
//         if (!match) {
//             print_hex("  NEON 结果:   ", neon_results[i]);
//             print_hex("  传统方法结果: ", manual_result);
//             all_match = false;
//         }
//     }
    
//     std::cout << std::endl;
//     std::cout << "总结: NEON 实现与传统方法" 
//               << (all_match ? "完全匹配" : "存在差异") << std::endl;
              
//     return 0;
// }