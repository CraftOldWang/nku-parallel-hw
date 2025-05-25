#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>

#ifdef F // 检查宏 F 是否被定义过，避免 undef 未定义的宏
#undef F
#endif


#include "ThreadPool.h"
#include "config.h"
#include <pthread.h>
using namespace std;
using namespace chrono;

// #define USING_SMALL

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2

auto thread_pool = new ThreadPool(THREAD_NUM); // 创建一个线程池，线程数可以根据需要调整
// 定义用于保护主要数据结构的互斥锁
pthread_mutex_t main_data_mutex = PTHREAD_MUTEX_INITIALIZER;
//大概会碰的就是 q.guesses吧？ 这个vector ，只有这个vector 是共享然后会写， 其他数据是安全的（要么不共享，要么只读）


int main()
{


    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    auto start_train = system_clock::now();
#ifdef _WIN32
    #ifdef USING_SMALL
    q.m.train(".\\guessdata\\small_Rockyou-singleLined-full.txt");
    #else
    q.m.train(".\\guessdata\\Rockyou-singleLined-full.txt");
    #endif
#else
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
#endif
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;


    
    // 加载一些测试数据
    unordered_set<std::string> test_set;
#ifdef _WIN32
    #ifdef USING_SMALL
    ifstream test_data(".\\guessdata\\small_Rockyou-singleLined-full.txt");
    #else
    ifstream test_data(".\\guessdata\\Rockyou-singleLined-full.txt");
    #endif
#else
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
#endif
    int test_count=0;
    string pw;
    while(test_data>>pw)
    {   
        test_count+=1;
        test_set.insert(pw);
        if (test_count>=1000000)
        {
            break;
        }
    }
    int cracked=0;

    q.init();

    cout << "here" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;
    // std::ofstream a("./files/results.txt");
    while (!q.priority.empty())
    {
        q.PopNext();

        pthread_mutex_lock(&main_data_mutex);
        q.total_guesses = q.guesses.size();
        pthread_mutex_unlock(&main_data_mutex);
        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " <<history + q.total_guesses << endl;
            // 这里读取什么的也许需要上锁？
            curr_num = q.total_guesses;

            // 在此处更改实验生成的猜测上限
            int generate_n=10000000;
            if (history + q.total_guesses > 10000000)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds"<< endl;
                cout << "Hash time:" << time_hash << "seconds"<<endl;
                cout << "Train time:" << time_train <<"seconds"<<endl;
                cout<<"Cracked:"<< cracked<<endl;
                break;
            }
        }
        // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
        // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
        if (curr_num > 1000000)
        {
            auto start_hash = system_clock::now();
            bit32 state[4];
            pthread_mutex_lock(&main_data_mutex); // hash 的时候，q.guesses不应该变动。
            for (string pw : q.guesses)
            {
                if (test_set.find(pw) != test_set.end()) {
                    cracked+=1;
                }
                // TODO：对于SIMD实验，将这里替换成你的SIMD MD5函数
                MD5Hash(pw, state);

                // 以下注释部分用于输出猜测和哈希，但是由于自动测试系统不太能写文件，所以这里你可以改成cout
                // a<<pw<<"\t";
                // for (int i1 = 0; i1 < 4; i1 += 1)
                // {
                //     a << std::setw(8) << std::setfill('0') << hex << state[i1];
                // }
                // a << endl;
            }
            pthread_mutex_unlock(&main_data_mutex);


            // 在这里对哈希所需的总时长进行计算
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            // 记录已经生成的口令总数
            history += curr_num;
            curr_num = 0;
            pthread_mutex_lock(&main_data_mutex);
            q.guesses.clear(); // hash 之后就清空，以免浪费多线程生成的猜测。(但是这样hash时间会变长？)
            //结果还是放回来了， 因为也许调用每个string 的析构很费时间。
            pthread_mutex_unlock(&main_data_mutex);
        }
    }
}
