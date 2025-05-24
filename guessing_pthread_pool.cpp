#include "PCFG.h"
#include "ThreadPool.h"
#include "ThreadPoolAsync.h"
#include <pthread.h>

using namespace std;

// 全局线程池实例定义
ThreadPool* global_thread_pool = nullptr;
pthread_mutex_t guess_mutex = PTHREAD_MUTEX_INITIALIZER;

// 全局异步任务跟踪
vector<int> active_task_groups;
pthread_mutex_t task_groups_mutex = PTHREAD_MUTEX_INITIALIZER;

// 初始化线程池
void init_thread_pool() {
    if (!global_thread_pool) {
        global_thread_pool = new ThreadPool();
    }
    
    // 初始化异步线程池
    init_async_thread_pool();
}

// 清理线程池
void cleanup_thread_pool() {
    if (global_thread_pool) {
        delete global_thread_pool;
        global_thread_pool = nullptr;
    }
    
    // 清理异步线程池
    cleanup_async_thread_pool();
}

// 添加任务组ID到活动列表
void add_active_task_group(int group_id) {
    pthread_mutex_lock(&task_groups_mutex);
    active_task_groups.push_back(group_id);
    pthread_mutex_unlock(&task_groups_mutex);
}

// 检查并清理已完成的任务组
void cleanup_completed_task_groups() {
    pthread_mutex_lock(&task_groups_mutex);
    auto it = active_task_groups.begin();
    while (it != active_task_groups.end()) {
        if (global_async_thread_pool->is_group_completed(*it)) {
            it = active_task_groups.erase(it);
        } else {
            ++it;
        }
    }
    pthread_mutex_unlock(&task_groups_mutex);
}

// 获取活动任务组数量
int get_active_task_groups_count() {
    pthread_mutex_lock(&task_groups_mutex);
    int count = active_task_groups.size();
    pthread_mutex_unlock(&task_groups_mutex);
    return count;
}

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    //BUGFIX 疑似没有清空？
    //BUGFIX测试发现， 真的没有清空啊
    //于是先清空一下
    // 不然由于没清空上次实验的PT，导致前面使用的PT，大多都是概率小，可能对应的value数也少？总之可能有影响
    priority.clear();

    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{
    // 清理已完成的任务组，这样可以释放内存和资源
    cleanup_completed_task_groups();
    
    // 更保守的并发控制策略，避免主线程跑得太快
    int thread_count = global_async_thread_pool->get_pool_size();
    int queue_size = global_async_thread_pool->get_queue_size();
    
    // 设置更保守的最大并发任务组数量
    int max_concurrent_groups = thread_count;  // 从3倍减少到1倍
    
    // 队列太满时，进一步降低并发度
    if (queue_size > thread_count * 5) {  // 从10倍改为5倍
        max_concurrent_groups = max(1, thread_count / 2);
    }
    // 队列接近空时，适度提高并发度
    else if (queue_size < thread_count / 2) {
        max_concurrent_groups = thread_count * 2;  // 从4倍减少到2倍
    }
    
    // 如果活动任务组数量超过阈值，等待一部分完成
    int active_groups = get_active_task_groups_count();
    while (active_groups > max_concurrent_groups) {
        // 更长的等待时间，确保有足够时间让任务完成
        usleep(5000);  // 5ms而不是动态计算
        cleanup_completed_task_groups();
        active_groups = get_active_task_groups_count();
        
        // 添加调试信息（可选）
        // cout << "等待任务完成: 活动组=" << active_groups << ", 最大允许=" << max_concurrent_groups << endl;
    }
    
    // 在Generate之前再次检查，确保系统稳定
    if (queue_size > thread_count * 8) {
        usleep(2000);  // 额外的2ms等待
    }
      // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测 (异步方式)
    Generate(priority.front());
    
    // 添加一个小的延迟，让异步任务有时间开始执行
    // 这可以避免主线程跑得太快导致的竞态条件
    usleep(100);  // 0.1ms的小延迟

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
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
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
          // 创建任务
        vector<AsyncTask> tasks;
        int total_values = pt.max_indices[0];
        
        // 每个任务至少处理5000个值，以减少线程调度开销
        const int MIN_CHUNK_SIZE = 5000;
        int num_threads = global_async_thread_pool->get_pool_size();
        int chunk_size;
        
        if (total_values < MIN_CHUNK_SIZE) {
            // 如果总值小于5000，则一个线程处理所有工作
            chunk_size = total_values;
        } else {
            // 计算每个线程至少处理5000个值的情况下需要多少线程
            int needed_threads = (total_values + MIN_CHUNK_SIZE - 1) / MIN_CHUNK_SIZE;
            
            // 如果需要的线程数少于可用线程数，每个线程处理一个完整的块
            if (needed_threads < num_threads) {
                chunk_size = MIN_CHUNK_SIZE;
            } else {
                // 否则平均分配，但保证每个任务至少有一定数量的工作
                chunk_size = max(1000, total_values / num_threads);
            }
        }
        
        for (int i = 0; i < total_values; i += chunk_size) {
            AsyncTask task;
            task.seg = a;
            task.prefix = "";
            task.start_idx = i;
            task.end_idx = min(i + chunk_size, total_values);
            task.result_vec = &guesses;
            task.counter = &total_guesses;
            task.mutex = &guess_mutex;
            task.is_single_segment = true;
            tasks.push_back(task);
        }
          // 异步提交任务，不等待完成
        if (tasks.size() > 0) {
            int group_id = global_async_thread_pool->submit_tasks_async(tasks);
            add_active_task_group(group_id);
        }
        
        // 注意：不再调用 wait_all_tasks() 以提高并发性能
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
          // 创建任务
        vector<AsyncTask> tasks;
        int total_values = pt.max_indices[pt.content.size() - 1];
        
        // 每个任务至少处理5000个值，以减少线程调度开销
        const int MIN_CHUNK_SIZE = 5000;
        int num_threads = global_async_thread_pool->get_pool_size();
        int chunk_size;
        
        if (total_values < MIN_CHUNK_SIZE) {
            // 如果总值小于5000，则一个线程处理所有工作
            chunk_size = total_values;
        } else {
            // 计算每个线程至少处理5000个值的情况下需要多少线程
            int needed_threads = (total_values + MIN_CHUNK_SIZE - 1) / MIN_CHUNK_SIZE;
            
            // 如果需要的线程数少于可用线程数，每个线程处理一个完整的块
            if (needed_threads < num_threads) {
                chunk_size = MIN_CHUNK_SIZE;
            } else {
                // 否则平均分配，但保证每个任务至少有一定数量的工作
                chunk_size = max(1000, total_values / num_threads);
            }
        }
        
        for (int i = 0; i < total_values; i += chunk_size) {
            AsyncTask task;
            task.seg = a;
            task.prefix = guess;
            task.start_idx = i;
            task.end_idx = min(i + chunk_size, total_values);
            task.result_vec = &guesses;
            task.counter = &total_guesses;
            task.mutex = &guess_mutex;
            task.is_single_segment = false;
            tasks.push_back(task);
        }
        
        // 异步提交任务，不等待完成
        int group_id = global_async_thread_pool->submit_tasks_async(tasks);
        add_active_task_group(group_id);
        
        // 对于multi-segment PT，不再同步等待，以允许最大程度的并发处理
    }
}