#include "PCFG.h"
#include "guessing_cuda.h"
#include "config.h"
#include <chrono>
#include <thread>

// 统一的缓冲区管理（无论是否使用线程池）
extern std::mutex gpu_buffer_mutex;
extern std::vector<char*> pending_gpu_buffers;

#include "ThreadPool.h"
extern std::unique_ptr<ThreadPool> thread_pool;
extern std::mutex main_data_mutex;
extern void async_gpu_task(AsyncGpuTask* task_data, PriorityQueue& q);

using namespace std;
using namespace chrono;

#ifdef TIME_COUNT
double time_gpu_kernel = 0;
double time_popnext_non_generate = 0;  // 新增：PopNext中非Generate部分的时间
double time_calprob = 0;  // 新增：CalProb函数的时间
#endif
// 添加最大任务数限制
const int MAX_PENDING_TASKS = 100;  // 限制最多100个异步任务 , 在 guess 生成猜测线程池中
void check_and_perform_hash();
void perform_hash_calculation(PriorityQueue& q, double& time_hash);
void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15
    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // 遍历所有已实例化的segment，累乘其概率
    SegmentLengthMaps* maps = SegmentLengthMaps::getInstance();

    for (int i = 0; i < pt.curr_indices.size(); i++) {
        // 🔥 确保不会超出范围，且不处理最后一个segment
        // if (i >= pt.content.size() - 1) {
        //     break;  // 安全退出
        // }
        const segment & cur_seg = maps->getSeginPQ(pt.content[i], *this);
        pt.prob *= cur_seg.ordered_freqs[pt.curr_indices[i]];
        pt.prob /= cur_seg.total_freq;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    //BUGFIX 疑似没有清空？
    //BUGFIX测试发现， 真的没有清空啊
    //于是先清空一下
    // 不然由于没清空上次实验的PT，导致前面使用的PT，大多都是概率小，可能对应的value数也少？总之可能有影响
    priority.clear();
    //  cout << priority.size() << endl;

    PTMaps *pt_maps = PTMaps::getInstance();
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {;
        pt.preterm_prob = float(m.preterm_freq[pt_maps->getPTID(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.insert(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{
#ifdef TIME_COUNT
    auto start_popnext = system_clock::now();
#endif
    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    auto front_iter = priority.begin();
    Generate(*front_iter);

#ifdef TIME_COUNT
    auto end_generate = system_clock::now();
    auto start_new_pts = system_clock::now();
#endif
    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = (*front_iter).NewPTs(*this);
    for (PT &pt : new_pts)
    {
#ifdef TIME_COUNT
    auto start_calprob = system_clock::now();
#endif
        // 计算概率
        CalProb(pt);
#ifdef TIME_COUNT
    auto end_calprob = system_clock::now();
    auto duration_calprob = duration_cast<microseconds>(end_calprob - start_calprob);
    time_calprob += double(duration_calprob.count()) * microseconds::period::num / microseconds::period::den;
#endif

        priority.insert(pt);

    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(front_iter);

#ifdef TIME_COUNT
    auto end_new_pts = system_clock::now();
    auto duration_new_pts = duration_cast<microseconds>(end_new_pts - start_new_pts);
    time_popnext_non_generate += double(duration_new_pts.count()) * microseconds::period::num / microseconds::period::den;
#endif
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs(PriorityQueue &q) const
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白

        // 开始遍历所有位置值大于等于init_pivot值的segment
        SegmentLengthMaps* maps = SegmentLengthMaps::getInstance();

        for (int i = pivot; i < curr_indices.size(); i++) {
            // 复制一份 curr_indices 到临时变量
            std::vector<int> temp_curr_indices = curr_indices;

            // 模拟加 1
            temp_curr_indices[i] += 1;

            const segment& model_seg = maps->getSeginPQ(content[i], q);
            size_t max_idx = model_seg.ordered_values.size();

            if (temp_curr_indices[i] < max_idx) {
                int temp_pivot = i;

                PT copy = *this;
                copy.pivot = temp_pivot;
                // 这里需要把 copy 的 curr_indices 替换成 temp_curr_indices，否则 copy 还是旧的状态
                copy.curr_indices = std::move(temp_curr_indices);

                res.emplace_back(std::move(copy));
            }
        }
        return res;
    }
    return res;
}



// 等待异步任务完成的函数
void wait_for_pending_tasks() {
    // cout << "i will wait" << endl;
    while (thread_pool->pending_tasks() >= MAX_PENDING_TASKS) {
#ifdef DEBUG
        printf("[DEBUG] ⏳ Waiting for tasks to complete... (thread_pool pending: %zu)\n", 
               thread_pool->pending_tasks());
#endif
        
        // 短暂等待，让CPU处理其他任务
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // cout <<"sleeping " << endl;

    }
    // cout<< "end wait" << endl;
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    // 1. 准备生成任务
    // 计算PT的概率，这里主要是给PT的概率进行初始化
#ifdef TIME_COUNT
    auto start_calprob = system_clock::now();
#endif
    CalProb(pt);
#ifdef TIME_COUNT
    auto end_calprob = system_clock::now();
    auto duration_calprob = duration_cast<microseconds>(end_calprob - start_calprob);
    time_calprob += double(duration_calprob.count()) * microseconds::period::num / microseconds::period::den;
#endif
    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可

    string prefix  = "";
    segment * a = nullptr;

    if (pt.content.size() == 1) {
        a = &pt.content[0];
    } else {
        a = &pt.content[pt.content.size() - 1];
        SegmentLengthMaps * maps = SegmentLengthMaps::getInstance();
        int seg_idx = 0;
        for (int idx : pt.curr_indices) // train.cpp 里写， curr_indices 是有最后一个seg的...不过没人说没有，好吧。
        {
            // 不能直接用。。。。线性查找改成hash，就这样了。
            const segment& cur_seg = pt.content[seg_idx];
            int cur_seg_type =  pt.content[seg_idx].type;
            int cur_seg_length = pt.content[seg_idx].length;
            prefix += maps->getSeginPQ(cur_seg, *this).ordered_values[idx];
            seg_idx += 1;
        }
    }

    // 2. 添加生成任务
    task_manager->add_task(a, prefix, *this);

    // 3. 如果达到一定量，以及一些要求， 就提交任务。
    if(task_manager->guesscount > GPU_BATCH_SIZE){
        // 🔥 关键：检查是否达到任务数限制
        if (thread_pool->pending_tasks() >= MAX_PENDING_TASKS) {
#ifdef DEBUG
            printf("[DEBUG] ⚠️ Max pending tasks (%d) reached, waiting... (current: %zu)\n", 
                   MAX_PENDING_TASKS, thread_pool->pending_tasks());
#endif

                // 等待任务数量小于一定值成
                wait_for_pending_tasks();
            }
            
            // 创建异步任务数据 (移动语义 把 task_manager 搬走)
            auto* async_task = new AsyncGpuTask(std::move(*task_manager));
            
            // 提交任务
            thread_pool->enqueue([async_task, this](){
                try {
                    async_gpu_task(async_task, *this); 
                } catch (...) {
                    std::cerr << "Async GPU task exception\n";
                }
            });


            // TaskManager已经被移动，重新创建一个新的
            task_manager = new TaskManager();

        }


    

}