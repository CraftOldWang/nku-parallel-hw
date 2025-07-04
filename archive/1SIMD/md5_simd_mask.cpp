#include "md5_simd.h"
#include <iomanip>
#include <assert.h>
#include <chrono>

using namespace std;
using namespace chrono;

const int BATCH_SIZE = 4;

// 废弃，最终没做

//TODO grok  说 口令 预处理成的块 数量不同， 可以使用掩码来进行每次的更新
// Claude 也说可以使用掩码， 还有什么位掩码  
// 		方案 1：使用完成掩码保存结果（推荐）
//	 	方案 2：使用位掩码保护已完成的状态（更高效但更复杂）



/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组；  
 * 这部分没有并行.... 
 * @param input 输入
 * @param[out] n_byte 用于给调用者传递额外的返回值，即最终Byte数组的长度
 * @return Byte消息数组
 */
Byte *StringProcess(string input, int *n_byte)
{
	// 将输入的字符串转换为Byte为单位的数组
	Byte *blocks = (Byte *)input.c_str();
	int length = input.length();

	// 计算原始消息长度（以比特为单位）
	int bitLength = length * 8;

	// paddingBits: 原始消息需要的padding长度（以bit为单位）
	// 对于给定的消息，将其补齐至length%512==448为止
	// 需要注意的是，即便给定的消息满足length%512==448，也需要再pad 512bits
	int paddingBits = bitLength % 512;
	if (paddingBits > 448)
	{
		paddingBits = 512 - (paddingBits - 448);
	}
	else if (paddingBits < 448)
	{
		paddingBits = 448 - paddingBits;
	}
	else if (paddingBits == 448)
	{
		paddingBits = 512;
	}

	// 原始消息需要的padding长度（以Byte为单位）
	int paddingBytes = paddingBits / 8;
	// 创建最终的字节数组
	// length + paddingBytes + 8:
	// 1. length为原始消息的长度（bits）
	// 2. paddingBytes为原始消息需要的padding长度（Bytes）
	// 3. 在pad到length%512==448之后，需要额外附加64bits的原始消息长度，即8个bytes
	int paddedLength = length + paddingBytes + 8;
	Byte *paddedMessage = new Byte[paddedLength];

	// 复制原始消息
	memcpy(paddedMessage, blocks, length);

	// 添加填充字节。填充时，第一位为1，后面的所有位均为0。
	// 所以第一个byte是0x80
	paddedMessage[length] = 0x80;							 // 添加一个0x80字节
	memset(paddedMessage + length + 1, 0, paddingBytes - 1); // 填充0字节

	// 添加消息长度（64比特，小端格式）
	for (int i = 0; i < 8; ++i)
	{
		// 特别注意此处应当将bitLength转换为uint64_t
		// 这里的length是原始消息的长度
		paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
	}

	// 验证长度是否满足要求。此时长度应当是512bit的倍数
	int residual = 8 * paddedLength % 512;
	// assert(residual == 0);

	// 在填充+添加长度之后，消息被分为n_blocks个512bit的部分
	*n_byte = paddedLength;
	return paddedMessage;
}




/**
 * MD5Hash_SIMD: 将 4 个输入字符串转换成MD5 
 * @param input 输入 , 长度为4 的string 数组
 * @param[out] state 用于给调用者传递额外的返回值，即最终的缓冲区，也就是MD5的结果
 * @return Byte消息数组
 */
void MD5Hash_SIMD_mask(string *input, uint32x4_t *state)
{

	Byte **paddedMessages = new Byte*[4];
	int *messageLength = new int[4];
	int *blockCounts = new int[4];

	for (int i = 0; i < 4; i += 1)
	{
		paddedMessages[i] = StringProcess(input[i], &messageLength[i]);
		blockCounts[i] = messageLength[i] / 64;
		// cout<<messageLength[i]<<endl;
		// assert(messageLength[i] == messageLength[0]);
	}
	// int n_blocks = messageLength[0] / 64;

    // 2. 找到最长的块数
    int max_blocks = blockCounts[0];
    for (int i = 1; i < 4; i++) {
        if (blockCounts[i] > max_blocks) {
            max_blocks = blockCounts[i];
        }
    }


	// bit32* state= new bit32[4];

	// state[0] = 0x67452301;
	// state[1] = 0xefcdab89;
	// state[2] = 0x98badcfe;
	// state[3] = 0x10325476;

    state[0] = vdupq_n_u32(0x67452301);
    state[1] = vdupq_n_u32(0xefcdab89);
    state[2] = vdupq_n_u32(0x98badcfe);
    state[3] = vdupq_n_u32(0x10325476);  

	bit32 tmp0[4];
	bit32 tmp1[4];
	bit32 tmp2[4];
	bit32 tmp3[4];

	// 逐block地更新state
	for (int i = 0; i < max_blocks; i += 1)
	{
		// bit32 x[16];
        uint32x4_t x[16];

		// 下面的处理，在理解上较为复杂
		// for (int i1 = 0; i1 < 16; ++i1)
		// {
		// 	x[i1] = (paddedMessages[4 * i1 + i * 64]) |
		// 			(paddedMessages[4 * i1 + 1 + i * 64] << 8) |
		// 			(paddedMessages[4 * i1 + 2 + i * 64] << 16) |
		// 			(paddedMessages[4 * i1 + 3 + i * 64] << 24);
		// }

		// 4.1 加载每个字符串的当前块
		for (int i1 = 0; i1 < 16; i1++) {
			uint32_t values[4] = {0}; // 默认值为 0 ; 如果某个 口令 已经算完了， 它的部分就填为0
			for (int j = 0; j < 4; j++) {
				if (i < blockCounts[j]) { // 检查是否超出块数
					values[j] = (paddedMessages[j][4 * i1 + i * 64]) |
								(paddedMessages[j][4 * i1 + 1 + i * 64] << 8) |
								(paddedMessages[j][4 * i1 + 2 + i * 64] << 16) |
								(paddedMessages[j][4 * i1 + 3 + i * 64] << 24);
				}
			}
			x[i1] = vld1q_u32(values); // 加载到 SIMD 向量
		}


		// bit32 a = state[0], b = state[1], c = state[2], d = state[3];
        uint32x4_t a = state[0],  b = state[1], c = state[2], d = state[3];

		auto start = system_clock::now();
		/* Round 1 */
		FF_SIMD(a, b, c, d, x[0], s11, 0xd76aa478);
		FF_SIMD(d, a, b, c, x[1], s12, 0xe8c7b756);
		FF_SIMD(c, d, a, b, x[2], s13, 0x242070db);
		FF_SIMD(b, c, d, a, x[3], s14, 0xc1bdceee);
		FF_SIMD(a, b, c, d, x[4], s11, 0xf57c0faf);
		FF_SIMD(d, a, b, c, x[5], s12, 0x4787c62a);
		FF_SIMD(c, d, a, b, x[6], s13, 0xa8304613);
		FF_SIMD(b, c, d, a, x[7], s14, 0xfd469501);
		FF_SIMD(a, b, c, d, x[8], s11, 0x698098d8);
		FF_SIMD(d, a, b, c, x[9], s12, 0x8b44f7af);
		FF_SIMD(c, d, a, b, x[10], s13, 0xffff5bb1);
		FF_SIMD(b, c, d, a, x[11], s14, 0x895cd7be);
		FF_SIMD(a, b, c, d, x[12], s11, 0x6b901122);
		FF_SIMD(d, a, b, c, x[13], s12, 0xfd987193);
		FF_SIMD(c, d, a, b, x[14], s13, 0xa679438e);
		FF_SIMD(b, c, d, a, x[15], s14, 0x49b40821);

		/* Round 2 */
		GG_SIMD(a, b, c, d, x[1], s21, 0xf61e2562);
		GG_SIMD(d, a, b, c, x[6], s22, 0xc040b340);
		GG_SIMD(c, d, a, b, x[11], s23, 0x265e5a51);
		GG_SIMD(b, c, d, a, x[0], s24, 0xe9b6c7aa);
		GG_SIMD(a, b, c, d, x[5], s21, 0xd62f105d);
		GG_SIMD(d, a, b, c, x[10], s22, 0x2441453);
		GG_SIMD(c, d, a, b, x[15], s23, 0xd8a1e681);
		GG_SIMD(b, c, d, a, x[4], s24, 0xe7d3fbc8);
		GG_SIMD(a, b, c, d, x[9], s21, 0x21e1cde6);
		GG_SIMD(d, a, b, c, x[14], s22, 0xc33707d6);
		GG_SIMD(c, d, a, b, x[3], s23, 0xf4d50d87);
		GG_SIMD(b, c, d, a, x[8], s24, 0x455a14ed);
		GG_SIMD(a, b, c, d, x[13], s21, 0xa9e3e905);
		GG_SIMD(d, a, b, c, x[2], s22, 0xfcefa3f8);
		GG_SIMD(c, d, a, b, x[7], s23, 0x676f02d9);
		GG_SIMD(b, c, d, a, x[12], s24, 0x8d2a4c8a);

		/* Round 3 */
		HH_SIMD(a, b, c, d, x[5], s31, 0xfffa3942);
		HH_SIMD(d, a, b, c, x[8], s32, 0x8771f681);
		HH_SIMD(c, d, a, b, x[11], s33, 0x6d9d6122);
		HH_SIMD(b, c, d, a, x[14], s34, 0xfde5380c);
		HH_SIMD(a, b, c, d, x[1], s31, 0xa4beea44);
		HH_SIMD(d, a, b, c, x[4], s32, 0x4bdecfa9);
		HH_SIMD(c, d, a, b, x[7], s33, 0xf6bb4b60);
		HH_SIMD(b, c, d, a, x[10], s34, 0xbebfbc70);
		HH_SIMD(a, b, c, d, x[13], s31, 0x289b7ec6);
		HH_SIMD(d, a, b, c, x[0], s32, 0xeaa127fa);
		HH_SIMD(c, d, a, b, x[3], s33, 0xd4ef3085);
		HH_SIMD(b, c, d, a, x[6], s34, 0x4881d05);
		HH_SIMD(a, b, c, d, x[9], s31, 0xd9d4d039);
		HH_SIMD(d, a, b, c, x[12], s32, 0xe6db99e5);
		HH_SIMD(c, d, a, b, x[15], s33, 0x1fa27cf8);
		HH_SIMD(b, c, d, a, x[2], s34, 0xc4ac5665);

		/* Round 4 */
		II_SIMD(a, b, c, d, x[0], s41, 0xf4292244);
		II_SIMD(d, a, b, c, x[7], s42, 0x432aff97);
		II_SIMD(c, d, a, b, x[14], s43, 0xab9423a7);
		II_SIMD(b, c, d, a, x[5], s44, 0xfc93a039);
		II_SIMD(a, b, c, d, x[12], s41, 0x655b59c3);
		II_SIMD(d, a, b, c, x[3], s42, 0x8f0ccc92);
		II_SIMD(c, d, a, b, x[10], s43, 0xffeff47d);
		II_SIMD(b, c, d, a, x[1], s44, 0x85845dd1);
		II_SIMD(a, b, c, d, x[8], s41, 0x6fa87e4f);
		II_SIMD(d, a, b, c, x[15], s42, 0xfe2ce6e0);
		II_SIMD(c, d, a, b, x[6], s43, 0xa3014314);
		II_SIMD(b, c, d, a, x[13], s44, 0x4e0811a1);
		II_SIMD(a, b, c, d, x[4], s41, 0xf7537e82);
		II_SIMD(d, a, b, c, x[11], s42, 0xbd3af235);
		II_SIMD(c, d, a, b, x[2], s43, 0x2ad7d2bb);
		II_SIMD(b, c, d, a, x[9], s44, 0xeb86d391);

		state[0] = vaddq_u32(state[0], a);
		state[1] = vaddq_u32(state[1], b);
		state[2] = vaddq_u32(state[2], c);
		state[3] = vaddq_u32(state[3], d);

		// 在每个 口令 结束之前， 用tmp存一下。 其对应的state
		if (i == blockCounts[0] - 1) {
			tmp0[0] = vgetq_lane_u32(state[0], 0);
			tmp1[0] = vgetq_lane_u32(state[1], 0);
			tmp2[0] = vgetq_lane_u32(state[2], 0);
			tmp3[0] = vgetq_lane_u32(state[3], 0);
		}
		if (i == blockCounts[1] - 1) {
			tmp0[1] = vgetq_lane_u32(state[0], 1);
			tmp1[1] = vgetq_lane_u32(state[1], 1);
			tmp2[1] = vgetq_lane_u32(state[2], 1);
			tmp3[1] = vgetq_lane_u32(state[3], 1);
		}
		if (i == blockCounts[2] - 1) {
			tmp0[2] = vgetq_lane_u32(state[0], 2);
			tmp1[2] = vgetq_lane_u32(state[1], 2);
			tmp2[2] = vgetq_lane_u32(state[2], 2);
			tmp3[2] = vgetq_lane_u32(state[3], 2);
		}
		if (i == blockCounts[3] - 1) {
			tmp0[3] = vgetq_lane_u32(state[0], 3);
			tmp1[3] = vgetq_lane_u32(state[1], 3);
			tmp2[3] = vgetq_lane_u32(state[2], 3);
			tmp3[3] = vgetq_lane_u32(state[3], 3);
		}
		

	}

	state[0] = vld1q_u32(tmp0);
	state[1] = vld1q_u32(tmp1);
	state[2] = vld1q_u32(tmp2);
	state[3] = vld1q_u32(tmp3);


	// 下面的处理，在理解上较为复杂
	// for (int i = 0; i < 4; i++)
	// {
	// 	uint32_t value = state[i];
	// 	state[i] = ((value & 0xff) << 24) |		 // 将最低字节移到最高位
	// 			   ((value & 0xff00) << 8) |	 // 将次低字节左移
	// 			   ((value & 0xff0000) >> 8) |	 // 将次高字节右移
	// 			   ((value & 0xff000000) >> 24); // 将最高字节移到最低位
	// }

	// 小端序大端序转换
	// 使用 NEON 指令一次性处理所有字节反转
	uint8x16_t bytes0 = vreinterpretq_u8_u32(state[0]);
	uint8x16_t bytes1 = vreinterpretq_u8_u32(state[1]);
	uint8x16_t bytes2 = vreinterpretq_u8_u32(state[2]);
	uint8x16_t bytes3 = vreinterpretq_u8_u32(state[3]);
	// vrev32q_u8 指令实现 32 位内的字节反转
	bytes0 = vrev32q_u8(bytes0);
	bytes1 = vrev32q_u8(bytes1);
	bytes2 = vrev32q_u8(bytes2);
	bytes3 = vrev32q_u8(bytes3);
	// 将反转后的字节重新解释为 uint32x4_t
	state[0] = vreinterpretq_u32_u8(bytes0);
	state[1] = vreinterpretq_u32_u8(bytes1);
	state[2] = vreinterpretq_u32_u8(bytes2);
	state[3] = vreinterpretq_u32_u8(bytes3);

	// 输出最终的hash结果
	// for (int i1 = 0; i1 < 4; i1 += 1)
	// {
	// 	cout << std::setw(8) << std::setfill('0') << hex << state[i1];
	// }
	// cout << endl;

	// 释放动态分配的内存
	// 实现SIMD并行算法的时候，也请记得及时回收内存！

	for (int i = 0; i < 4; i++) {
		delete[] paddedMessages[i];
	}
	delete[] paddedMessages;
	delete[] messageLength;
	delete[] blockCounts;
}


