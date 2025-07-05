#ifndef GUESSING_CUDA_H
#define GUESSING_CUDA_H
#include <vector>
#include "PCFG.h"
#include <numeric> // for std::accumulate
#include <device_launch_parameters.h>
#include <cuda_runtime.h>


#include "config.h"


#include "ThreadPool.h"
#include <mutex>
#include <atomic>
// 各种 外部变量。
extern std::atomic<int> pending_task_count;  // 未完成的异步任务数量
extern mutex main_data_mutex;
extern mutex gpu_buffer_mutex;
extern std::vector<char*> pending_gpu_buffers;  // 等待释放的GPU缓冲区指针
extern std::unique_ptr<ThreadPool> thread_pool;
extern PriorityQueue q;


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            fprintf(stderr, "\033[1;31m[CUDA ERROR]\033[0m %s:%d: %s (%d)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err__), err__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)




// GPU上用于查找的 ordered_values 数据结构
struct GpuOrderedValuesData{
    char* letter_all_values;// 把各个segment 的ordered_values展平
    char* digit_all_values;
    char* symbol_all_values;

    int* letter_value_offsets; // 每个ordered_value在letter_all_values中的起始
    int* digits_value_offsets; // 每个ordered_value在digit_all_values中的起始位置
    int* symbol_value_offsets; // 每个ordered_value在symbol_all_values中的起始

    int* letter_seg_offsets; // 每个letter segment第一个ordered_value在value_offsets中的是哪个
    int* digit_seg_offsets; // 每个digit segment第一个ordered_value在value_offsets中的是哪个
    int* symbol_seg_offsets; // 每个symbol segment第一个ordered_value在value_offsets
};


// 给GPU用的 一批任务 。包含了生成guess所需的数据
struct Taskcontent{ 
    int* seg_types; // 1: letter, 2: digit, 3: symbol (0:未设置)
    int* seg_ids;   // 对应在 model 里的 三个vector 的下标
    int* seg_lens;
    const char* prefixs;
    int* prefix_offsets;
    int* prefix_lens; // 每个prefix的长度
    int* seg_value_counts; // 每个segment的value数量
    int* cumulative_guess_offsets; // 累积guess偏移数组，用于二分查找优化

    int* output_offsets;  // 每个task在输出buffer中的起始位置

    int taskcount;
    int guesscount; // 到10_0000了就会丢给核函数去执行
};

// 独立的映射表管理类（单例模式）
class SegmentLengthMaps {
private:
    static SegmentLengthMaps* instance;
    unordered_map<int, int> letter_length_to_id;
    unordered_map<int, int> digit_length_to_id;
    unordered_map<int, int> symbol_length_to_id;
    bool initialized = false;
    
public:
    static SegmentLengthMaps* getInstance() {
        if (instance == nullptr) {
            instance = new SegmentLengthMaps();
        }
        return instance;
    }
    
    void init(PriorityQueue& q);
    
    int getLetterID(int length) const { return letter_length_to_id.at(length); }
    int getDigitID(int length) const { return digit_length_to_id.at(length); }
    int getSymbolID(int length) const { return symbol_length_to_id.at(length); }
    int getID(const segment& seg) const {
        switch (seg.type)
        {
        case 1:
            return letter_length_to_id.at(seg.length); 
            break;
        case 2:
            return digit_length_to_id.at(seg.length);
            break;
        case 3:
            return symbol_length_to_id.at(seg.length);
            break;
        default:
            throw "undefined_segment_error";
            break;
        }
    }
    const segment& getSeginPQ(const segment& seg, PriorityQueue & q) const {
        switch (seg.type)
        {
        case 1:
            return  q.m.letters[getID(seg)]; 
            break;
        case 2:
            return q.m.digits[getID(seg)]; 
            break;
        case 3:
            return q.m.symbols[getID(seg)]; 
            break;
        default:
            throw "undefined_segment_error";
            break;
        }
    }
};

// PT映射表管理类（单例模式）
class PTMaps {
private:
    static PTMaps* instance;
    unordered_map<string, int> pt_signature_to_id;  // PT签名到ID的映射
    bool initialized = false;
    
    // 生成PT的唯一签名
    string generatePTSignature(const PT& pt) const {
        string signature;
        for (const auto& seg : pt.content) {
            signature += to_string(seg.type) + "_" + to_string(seg.length) + "|";
        }
        return signature;
    }
    
public:
    static PTMaps* getInstance() {
        if (instance == nullptr) {
            instance = new PTMaps();
        }
        return instance;
    }
    
    void init(PriorityQueue& q) {
        if (initialized) return;
        
        // 构建PT签名到索引的映射
        for (int i = 0; i < q.m.preterminals.size(); i++) {
            string signature = generatePTSignature(q.m.preterminals[i]);
            pt_signature_to_id[signature] = i;
        }
        
        initialized = true;
        
        #ifdef DEBUG
        cout << "PTMaps initialized with " << pt_signature_to_id.size() << " PT mappings" << endl;
        #endif
    }
    
    // 根据PT获取其在模型中的ID
    int getPTID(const PT& pt) const {
        string signature = generatePTSignature(pt);
        auto it = pt_signature_to_id.find(signature);
        if (it != pt_signature_to_id.end()) {
            return it->second;
        }
        throw std::runtime_error("PT not found in mapping: " + signature);
    }
    
    // 根据PT获取其在模型中的引用
    const PT& getPTInModel(const PT& pt, PriorityQueue& q) const {
        int id = getPTID(pt);
        return q.m.preterminals[id];
    }
    
    // 根据PT获取其在模型中的指针
    PT* getPTPtr(const PT& pt, PriorityQueue& q) const {
        int id = getPTID(pt);
        return &q.m.preterminals[id];
    }
    
    // 获取PT的频率
    int getPTFreq(const PT& pt, PriorityQueue& q) const {
        int id = getPTID(pt);
        auto freq_it = q.m.preterm_freq.find(id);
        if (freq_it != q.m.preterm_freq.end()) {
            return freq_it->second;
        }
        return 0;
    }
    
    // 获取PT的概率
    float getPTProb(const PT& pt, PriorityQueue& q) const {
        int freq = getPTFreq(pt, q);
        return static_cast<float>(freq) / q.m.total_preterm;
    }
    
    // 检查PT是否存在于模型中
    bool containsPT(const PT& pt) const {
        string signature = generatePTSignature(pt);
        return pt_signature_to_id.find(signature) != pt_signature_to_id.end();
    }
    
    // 获取映射表大小
    size_t size() const {
        return pt_signature_to_id.size();
    }
    
    // 清理资源
    void cleanup() {
        pt_signature_to_id.clear();
        initialized = false;
    }
    
    // 调试：打印所有映射
    void printMappings() const {
        cout << "=== PT Mappings ===" << endl;
        for (const auto& pair : pt_signature_to_id) {
            cout << "Signature: " << pair.first << " -> ID: " << pair.second << endl;
        }
        cout << "===================" << endl;
    }
};

// 静态成员定义（需要在.cu文件中）
// PTMaps* PTMaps::instance = nullptr;



class TaskManager{
public:
    vector<int> seg_types;
    vector<int> seg_ids;
    vector<int> seg_lens;
    vector<string> prefixs;
    vector<int> prefix_lens; // 每个prefix的长度
    vector<int> seg_value_count; // 每个seg 有多少 value （其实只会有所谓最后一个seg的， 就是后面一个seg生成相应数量guess）
    int taskcount;
    int guesscount; // 到10_0000了就会丢给核函数去执行

    TaskManager(): taskcount(0), guesscount(0){}// vector 自己会调自己的构造..
    
    // 移动构造函数
    TaskManager(TaskManager&& other) noexcept 
        : seg_types(std::move(other.seg_types))
        , seg_ids(std::move(other.seg_ids))
        , seg_lens(std::move(other.seg_lens))
        , prefixs(std::move(other.prefixs))
        , prefix_lens(std::move(other.prefix_lens))
        , seg_value_count(std::move(other.seg_value_count))
        , taskcount(other.taskcount)
        , guesscount(other.guesscount) {
        other.taskcount = 0;
        other.guesscount = 0;
    }

    // 禁用拷贝构造和赋值（强制使用移动语义）
    TaskManager(const TaskManager&) = delete;
    TaskManager& operator=(const TaskManager&) = delete;
    void add_task(const segment* seg, string& prefix, PriorityQueue& q);

    // 统一函数签名，都接受外部缓冲区指针
    void launch_gpu_kernel(vector<string_view>& guesses, PriorityQueue& q, char*& h_guess_buffer);
    
    // 计算所需的缓冲区大小
    void clean();
    void print();
};


struct AsyncGpuTask {
    TaskManager task_manager;  // 直接移动，不用指针
    
    // 移动构造函数
    AsyncGpuTask(TaskManager&& tm) 
        : task_manager(std::move(tm)){}
};

void async_gpu_task(AsyncGpuTask* task_data, PriorityQueue& q);
void sync_gpu_task(AsyncGpuTask* task_data, PriorityQueue& q);




class AsyncGpuPipeline {
public:

    struct AsyncTaskData {
        // 原有数据
        TaskManager task_manager;
        vector<string_view> local_guesses;
        char* gpu_buffer;
        
        // ⚠️ 添加：保证数据生命周期
        std::string all_prefixes;           // 保存连接的前缀字符串
        std::vector<int> res_offset;        // 保存结果偏移量
        std::vector<int> cumulative_offsets; // 保存累积偏移量
        Taskcontent task_content;
        // ⚠️ 添加：错误状态管理
        std::atomic<bool> has_error{false};

        // CUDA异步相关
        cudaStream_t compute_stream;
        
        // GPU内存指针（用于异步释放）
        char* temp_prefixs;
        int* d_seg_types;
        int* d_seg_ids;
        int* d_seg_lens;
        int* d_prefix_offsets;
        int* d_prefix_lens;
        int* d_seg_value_counts;
        int* d_cumulative_guess_offsets;
        int* d_output_offsets;
        Taskcontent* d_tasks;
        char* d_guess_buffer;
        
        // CPU内存指针
        int* h_prefix_offsets;
        
        // 结果相关
        size_t result_len;
        
        AsyncTaskData(TaskManager&& tm);
        
        ~AsyncTaskData();
    };
public:
    static void launch_async_pipeline(TaskManager tm, PriorityQueue& q);
};
// 流水线阶段函数
void prepare_gpu_data_stage(AsyncGpuPipeline::AsyncTaskData& data);
void launch_kernel_stage(AsyncGpuPipeline::AsyncTaskData& data);
void submit_memory_copy_task(AsyncGpuPipeline::AsyncTaskData* data, PriorityQueue& q);
void process_strings_stage(AsyncGpuPipeline::AsyncTaskData& data);
void merge_results_stage(AsyncGpuPipeline::AsyncTaskData& data);
// 清理函数
void cleanup_stage(AsyncGpuPipeline::AsyncTaskData& data);
void synchronous_cleanup(AsyncGpuPipeline::AsyncTaskData& data);






// 🔥 GPU任务的不同阶段
enum class GpuTaskStage {
    PREPARE_DATA,    // 准备数据并拷贝到GPU
    LAUNCH_KERNEL,   // 启动GPU kernel
    COPY_RESULTS,    // 拷贝结果回CPU
    PROCESS_STRINGS, // 处理字符串
    MERGE_RESULTS    // 合并到全局队列
};

// 🔥 GPU任务数据结构（包含状态）
struct StagedGpuTask {
    TaskManager task_manager;
    vector<string_view> local_guesses;
    char* gpu_buffer = nullptr;
    
    // 🔥 当前阶段和状态
    GpuTaskStage current_stage = GpuTaskStage::PREPARE_DATA;
    bool has_error = false;

    // ⚠️ 添加：保证数据生命周期  阶段间的数据
    std::string all_prefixes;           // 保存连接的前缀字符串
    std::vector<int> res_offset;        // 保存结果偏移量
    std::vector<int> cumulative_offsets; // 保存累积偏移量
    size_t result_len = 0;

    // CUDA资源
    cudaStream_t compute_stream = nullptr;
    
    // GPU内存指针（用于分阶段释放）
    char* temp_prefixs = nullptr;
    int* d_seg_types = nullptr;
    int* d_seg_ids = nullptr;
    int* d_seg_lens = nullptr;
    int* d_prefix_offsets = nullptr;
    int* d_prefix_lens = nullptr;
    int* d_seg_value_counts = nullptr;
    int* d_cumulative_guess_offsets = nullptr;
    int* d_output_offsets = nullptr;
    Taskcontent* d_tasks = nullptr;
    char* d_guess_buffer = nullptr;
    
    // CPU内存指针
    int* h_prefix_offsets = nullptr;

    StagedGpuTask(TaskManager&& tm) : task_manager(std::move(tm)) {
        // 创建CUDA流
        CUDA_CHECK(cudaStreamCreate(&compute_stream));
    }
    
    ~StagedGpuTask() {
        if (compute_stream) {
            cudaStreamDestroy(compute_stream);
        }
    }
    void cleanup_gpu_resources();

};

// 🔥 分阶段GPU任务处理函数
void process_staged_gpu_task(StagedGpuTask* task, PriorityQueue& q);

// 🔥 各个阶段的函数声明
void stage_prepare_data(StagedGpuTask* task, PriorityQueue& q);
void stage_launch_kernel(StagedGpuTask* task, PriorityQueue& q);
void stage_copy_results(StagedGpuTask* task, PriorityQueue& q);
void stage_process_strings(StagedGpuTask* task, PriorityQueue& q);
void stage_merge_results(StagedGpuTask* task, PriorityQueue& q);
void cleanup_staged_task(StagedGpuTask* task);





// 生成猜测的 kernal 函数 。生成的猜测放到 d_guess_buffer 上
__global__ void generate_guesses_kernel(
    GpuOrderedValuesData* gpu_data,
    Taskcontent* d_tasks,
    char* d_guess_buffer
);

// 全局变量声明
extern GpuOrderedValuesData* gpu_data;
extern TaskManager* task_manager;

// 初始化函数、清理函数声明

// 初始化gpu上用于查找的数据 (segment表)
void init_gpu_ordered_values_data(GpuOrderedValuesData*& d_gpu_data, PriorityQueue& q);
// 清理gpu上用来查找的数据 (segment表)
void clean_gpu_ordered_values_data(GpuOrderedValuesData*& d_gpu_data);
// 清理 segment表 和 taskmanager (清理两个全局变量) 
void cleanup_global_cuda_resources();

#endif