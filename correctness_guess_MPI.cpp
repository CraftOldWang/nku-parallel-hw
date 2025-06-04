#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <mpi.h>
using namespace std;
using namespace chrono;

// 编译指令如下
// mpicxx correctness_guess_MPI.cpp train.cpp guessing_MPI.cpp md5.cpp -o main_mpi
// mpicxx correctness_guess_MPI.cpp train.cpp guessing_MPI.cpp md5.cpp -o main_mpi -O1
// mpicxx correctness_guess_MPI.cpp train.cpp guessing_MPI.cpp md5.cpp -o main_mpi -O2

// MPI消息标签定义
#define TAG_TASK 1
#define TAG_RESULT 2
#define TAG_TERMINATE 3

// 工作进程函数声明
void worker_process();

int main()
{
    // 初始化MPI
    MPI_Init(NULL, NULL);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        // 主进程执行原有逻辑
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;


    
    // 加载一些测试数据
    unordered_set<std::string> test_set;
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
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
        q.total_guesses = q.guesses.size();
        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " <<history + q.total_guesses << endl;
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

            // 在这里对哈希所需的总时长进行计算
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            // 记录已经生成的口令总数
            history += curr_num;
            curr_num = 0;            
            q.guesses.clear();
        }
    }
    
    // 发送终止信号给所有工作进程
    for (int i = 1; i < size; i++) {
        int terminate_signal = -1;
        MPI_Send(&terminate_signal, 1, MPI_INT, i, TAG_TERMINATE, MPI_COMM_WORLD);
    }
    
    } else {
        // 工作进程
        worker_process();
    }
    
    MPI_Finalize();
    return 0;
}

// 工作进程函数实现
void worker_process() {
    while (true) {
        // 接收任务信号
        int task_signal;
        MPI_Status status;
        MPI_Recv(&task_signal, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        
        if (status.MPI_TAG == TAG_TERMINATE) {
            break; // 收到终止信号，退出循环
        }
          if (status.MPI_TAG == TAG_TASK) {
            // 接收任务数据
            int prefix_len, values_count, start_idx, end_idx;
            MPI_Recv(&prefix_len, 1, MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD, &status);
            
            string prefix = "";
            if (prefix_len > 0) {
                char* prefix_buffer = new char[prefix_len + 1];
                MPI_Recv(prefix_buffer, prefix_len, MPI_CHAR, 0, TAG_TASK, MPI_COMM_WORLD, &status);
                prefix_buffer[prefix_len] = '\0';
                prefix = string(prefix_buffer);
                delete[] prefix_buffer;
            }
            
            MPI_Recv(&values_count, 1, MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD, &status);
            MPI_Recv(&start_idx, 1, MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD, &status);
            MPI_Recv(&end_idx, 1, MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD, &status);
            
            // 接收values数组
            vector<string> values(values_count);
            for (int i = 0; i < values_count; i++) {
                int value_len;
                MPI_Recv(&value_len, 1, MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD, &status);
                char* value_buffer = new char[value_len + 1];
                MPI_Recv(value_buffer, value_len, MPI_CHAR, 0, TAG_TASK, MPI_COMM_WORLD, &status);
                value_buffer[value_len] = '\0';
                values[i] = string(value_buffer);
                delete[] value_buffer;
            }
            
            // 生成猜测
            vector<string> local_guesses;
            for (int i = start_idx; i < end_idx; i++) {
                string guess = prefix + values[i];
                local_guesses.push_back(guess);
            }
            
            // 发送结果回主进程
            int result_count = local_guesses.size();
            MPI_Send(&result_count, 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
            
            for (const string& guess : local_guesses) {
                int guess_len = guess.length();
                MPI_Send(&guess_len, 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
                MPI_Send(guess.c_str(), guess_len, MPI_CHAR, 0, TAG_RESULT, MPI_COMM_WORLD);
            }
        }
    }
}
