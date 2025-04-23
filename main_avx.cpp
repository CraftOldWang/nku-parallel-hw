#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include "md5_avx.h"  
#include <iomanip>
#include <immintrin.h> 
#include <filesystem> 

using namespace std;
using namespace chrono;
namespace fs = std::filesystem;

int GENERATE_N = 10000000;
int NUM_PER_HASH = 1000000;

// 编译指令
// g++ main_avx.cpp train.cpp guessing.cpp md5.cpp md5_avx.cpp -o main_avx -mavx2 -O3

int main()
{
    double time_hash = 0;  
    double time_guess = 0; 
    double time_train = 0; 
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train(".\\guessdata\\Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "here" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    int history = 0;
    
    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();
        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            // 在此处更改实验生成的猜测上限
            int generate_n=GENERATE_N;
            if (history + q.total_guesses > GENERATE_N)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << " seconds" << endl;
                cout << "Hash time:" << time_hash << " seconds" << endl;
                cout << "Train time:" << time_train << " seconds" << endl;


                // 确保结果目录存在
                if (!fs::exists("results")) {
                    fs::create_directory("results");
                }
                
                // 打开文件以附加模式写入结果
                ofstream result_file("results/avx_result.txt", ios::app);
                if (result_file.is_open()) {
                    // 获取当前时间作为记录标识
                    auto now = system_clock::now();
                    auto now_time = system_clock::to_time_t(now);
                    
                    // 写入结果
                    result_file << "--- 实验结果 [" << std::ctime(&now_time) << "] ---" << endl;
                    result_file << "GENERATE_N: " << GENERATE_N << endl;
                    result_file << "NUM_PER_HASH: " << NUM_PER_HASH << endl;
                    result_file << "Guess time: " << (time_guess - time_hash) << " seconds" << endl;
                    result_file << "Hash time: " << time_hash << " seconds" << endl;
                    result_file << "Train time: " << time_train << " seconds" << endl;
                    result_file << "Total time: " << time_guess << " seconds" << endl;
                    result_file << "-----------------------------" << endl << endl;
                    
                    result_file.close();
                    cout << "结果已保存到 results/avx_result.txt" << endl;
                } else {
                    cerr << "无法打开结果文件进行写入！" << endl;
                }
                break;
            }
        }
        
        if (curr_num > 1000000)
        {
            cout << "Start performing the hash calculation..." << endl;
            auto start_hash = system_clock::now();
            
            
            for(size_t i = 0; i < q.guesses.size(); i += 8) {
                string passwords[8] = {"", "", "", "", "", "", "", ""};
                
                // cout << "batch processing " << (i/8) << " (indexes " << i << ")" << endl;
                
                // 填充密码数组（考虑边界情况）
                for (int j = 0; j < 8 && (i + j) < q.guesses.size(); ++j) {
                    passwords[j] = q.guesses[i + j];
                }
                if (i >= 100000){
                    break;
                }
                // 使用AVX版本计算MD5
                __m256i state[4]; // 8个密码的状态向量
                MD5Hash_AVX(passwords, state);
            }
            

            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
    
    return 0;
}