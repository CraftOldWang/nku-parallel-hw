#include "PCFG.h"

#ifdef F // 检查宏 F 是否被定义过，避免 undef 未定义的宏
#undef F
#endif

#include "ThreadPool.h"
#include <pthread.h>
#include "config.h"

using namespace std;

extern ThreadPool *thread_pool;
// 中间结果（向量、计数器）（不需要传参） 、
//  最终结果向量、最终计数器的引用（指针） 、 seg指针、前缀()、是否单seg、
// 用于保护主要数据结构的互斥锁，在main中访问q.guesses和q.total_guesses时使用
extern pthread_mutex_t main_data_mutex;


struct ThreadArgs {
    vector<string>* result_vec;   // 存储生成的猜测字符串
    segment* seg;
    string prefix;               // 前缀 (对于多segment情况)
    bool is_single_segment; // 是否单segment
    int start_idx;                // 起始索引
    int end_idx;                  // 结束索引(不包含)
};
void parallel_task(ThreadArgs* args){ // 也许为了保证threadargs不会随generate作用域而寄，应该在堆上弄？
    vector<string> local_result; // 也许 args-> 指针访问，放循环里会拖慢速度？
    // 也许 main 里面， 进行 guess 的清空， 访问等等的时候， 也需要使用互斥锁？
    local_result.reserve(args->end_idx - args->start_idx);
    if ( args->is_single_segment){
        for(int i = args->start_idx; i < args->end_idx; ++i){
            // string temp = args->seg->ordered_values[i];
            string temp = args->seg->ordered_values[i];
            local_result.emplace_back(temp);
        }
    } else {
        for(int i = args->start_idx; i < args->end_idx; ++i){
            string temp = args->prefix + args->seg->ordered_values[i];
            local_result.emplace_back(temp);
        }
    }   

    pthread_mutex_lock(&main_data_mutex);
    args->result_vec->insert(args->result_vec->end(), local_result.begin(), local_result.end());
    pthread_mutex_unlock(&main_data_mutex);
    delete args;
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
    // cout << m.ordered_pts.size() << endl;
    //BUGFIX 疑似没有清空？
    //BUGFIX测试发现， 真的没有清空啊
    //于是先清空一下
    // 不然由于没清空上次实验的PT，导致前面使用的PT，大多都是概率小，可能对应的value数也少？总之可能有影响
    priority.clear();
    //  cout << priority.size() << endl;

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

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

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
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        // for (int i = 0; i < pt.max_indices[0]; i += 1)
        // {
        //     string guess = a->ordered_values[i];
        //     // cout << guess << endl;
        //     guesses.emplace_back(guess);
        //     total_guesses += 1;
        // }

        // 每8000个作为一个 任务， 然后如果大于8000*线程数，则平均分给各个线程？
        int total_values = pt.max_indices[0];
        if (total_values <= 8000) {
            //只分给一个线程
            auto *arg = new ThreadArgs(); // 也许需要在堆上？ 然后在任务结束时释放？
            arg->result_vec = &guesses;
            arg->seg = a;
            arg->is_single_segment = true;
            arg->prefix = "";
            arg->start_idx = 0;
            arg->end_idx = total_values;
            thread_pool->enqueue(parallel_task,arg); // 如果值传递。。。传结构体，复制开销大吗？
        } else if (total_values >= 8000* THREAD_NUM) { // 这个8000 也可以当成超参数
            int task_per_thread = total_values / THREAD_NUM;
            int remainder = total_values - task_per_thread* THREAD_NUM;
            for (int i = 0 ; i < THREAD_NUM;i++) {
                auto *arg = new ThreadArgs();
                arg->result_vec = &guesses;
                arg->seg = a;
                arg->is_single_segment = true;
                arg->prefix = "";
                arg->start_idx = task_per_thread* i + min(i, remainder);
                arg->end_idx = task_per_thread*(i+1) + min(i+1, remainder);
                thread_pool->enqueue(parallel_task, arg); // &args[i] 等价于 &(args[i])
            }
        } else {//  8000 < total_values < 8000* 线程数
            //每个线程分配8000 ，分完就不分了。
            int thread_to_use = total_values / 8000;
            int remainder = total_values - 8000* thread_to_use;
            if (remainder!=0) {
                thread_to_use += 1;
            }
            for (int i = 0 ; i < thread_to_use;i++) {
                auto *arg = new ThreadArgs();
                arg->result_vec = &guesses;
                arg->seg = a;
                arg->is_single_segment = true;
                arg->prefix = "";
                arg->start_idx = 8000* i;
                // 如果有剩余，最后一个没有分配到8000 ， 而是会更少
                arg->end_idx = (remainder != 0 && i == thread_to_use-1) ? total_values :8000*(i+1) ;
                thread_pool->enqueue(parallel_task, arg); // &args[i] 等价于 &(args[i])
            }
        }

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
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        // for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        // {
        //     string temp = guess + a->ordered_values[i];
        //     // cout << temp << endl;
        //     guesses.emplace_back(temp);
        //     total_guesses += 1;
        // }

        // 每8000个作为一个 任务， 然后如果大于8000*线程数，则平均分给各个线程？
        int total_values = pt.max_indices[pt.content.size() - 1];
        if (total_values <= 8000) {
            //只分给一个线程
            auto *arg = new ThreadArgs(); // 也许需要在堆上？ 然后在任务结束时释放？
            arg->result_vec = &guesses;
            arg->seg = a;
            arg->is_single_segment = false;
            arg->prefix = guess;
            arg->start_idx = 0;
            arg->end_idx = total_values;
            thread_pool->enqueue(parallel_task,arg); // 如果值传递。。。传结构体，复制开销大吗？
        } else if (total_values >= 8000* THREAD_NUM) { // 这个8000 也可以当成超参数
            int task_per_thread = total_values / THREAD_NUM;
            int remainder = total_values - task_per_thread* THREAD_NUM;
            for (int i = 0 ; i < THREAD_NUM;i++) {
                auto arg = new ThreadArgs();
                arg->result_vec = &guesses;
                arg->seg = a;
                arg->is_single_segment = false;
                arg->prefix = guess;
                arg->start_idx = task_per_thread* i + min(i, remainder);
                arg->end_idx = task_per_thread*(i+1) + min(i+1, remainder);
                thread_pool->enqueue(parallel_task, arg); // &args[i] 等价于 &(args[i])
            }
        } else {//  8000 < total_values < 8000* 线程数
            //每个线程分配8000 ，分完就不分了。
            int thread_to_use = total_values / 8000;
            int remainder = total_values - 8000* thread_to_use;
            if (remainder!=0) {
                thread_to_use += 1;
            }
            for (int i = 0 ; i < thread_to_use;i++) {
                auto arg = new ThreadArgs();
                arg->result_vec = &guesses;
                arg->seg = a;
                arg->is_single_segment = false;
                arg->prefix = guess;
                arg->start_idx = 8000* i;
                // 如果有剩余，最后一个没有分配到8000 ， 而是会更少
                arg->end_idx = (remainder != 0 && i == thread_to_use-1) ? total_values :8000*(i+1) ;
                thread_pool->enqueue(parallel_task, arg); // &args[i] 等价于 &(args[i])
            }
        }

    }
}

