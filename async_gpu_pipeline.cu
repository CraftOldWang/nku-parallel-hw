#include "guessing_cuda.h"
#include "PCFG.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <numeric>
#include <climits>

// 必要的extern声明
extern std::unique_ptr<ThreadPool> thread_pool;
extern std::mutex main_data_mutex;
extern std::mutex gpu_buffer_mutex;
extern std::vector<char*> pending_gpu_buffers;
extern GpuOrderedValuesData* gpu_data;



// 安全的CUDA错误检查（用于回调函数）
#define CUDA_CHECK_CALLBACK(call, cleanup_action) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            std::cerr << "CUDA error in callback: " << cudaGetErrorString(err__) << std::endl; \
            cleanup_action; \
            return; \
        } \
    } while (0)

AsyncGpuPipeline::AsyncTaskData::AsyncTaskData(TaskManager&& tm)
    : task_manager(std::move(tm)), gpu_buffer(nullptr) {
    // 创建CUDA streams和events
    cudaStreamCreate(&compute_stream);

    // 初始化指针
    temp_prefixs = nullptr;
    d_seg_types = nullptr;
    d_seg_ids = nullptr;
    d_seg_lens = nullptr;
    d_prefix_offsets = nullptr;
    d_prefix_lens = nullptr;
    d_seg_value_counts = nullptr;
    d_cumulative_guess_offsets = nullptr;
    d_output_offsets = nullptr;
    d_tasks = nullptr;
    d_guess_buffer = nullptr;
    h_prefix_offsets = nullptr;
    result_len = 0;
}

AsyncGpuPipeline::AsyncTaskData::~AsyncTaskData(){
            // 清理CUDA资源
            if (compute_stream) cudaStreamDestroy(compute_stream);
}



void sync_gpu_task(AsyncGpuTask* task_data, PriorityQueue& q) {
#ifdef DEBUG
    printf("[DEBUG] 🎯 sync_gpu_task: Starting synchronous GPU task\n");
#endif

    try {
        
        // 创建临时数据结构
        AsyncGpuPipeline::AsyncTaskData data(std::move(task_data->task_manager));
        
        // 阶段1: 准备GPU数据
        prepare_gpu_data_stage(data);

        
        // 阶段2: 启动GPU kernel
        launch_kernel_stage(data);
        
        // 🔥 阶段3: 同步等待GPU完成
        cudaError_t sync_err = cudaStreamSynchronize(data.compute_stream);
        if (sync_err != cudaSuccess) {
            throw std::runtime_error("GPU kernel execution failed: " + std::string(cudaGetErrorString(sync_err)));
        }
        
        // 阶段4: 同步拷贝结果回CPU
        CUDA_CHECK(cudaMemcpy(data.gpu_buffer, data.d_guess_buffer, 
                             data.result_len * sizeof(char), cudaMemcpyDeviceToHost));
        
        // 阶段5: CPU处理
        process_strings_stage(data);
        
        // 阶段6: 合并结果到全局队列
        merge_results_stage(data);
        
        // 阶段7: 同步清理所有资源
        synchronous_cleanup(data);
        
#ifdef DEBUG
        printf("[DEBUG] ✅ sync_gpu_task: GPU task completed successfully\n");
#endif

    } catch (const std::exception& e) {
        std::cerr << "Sync GPU task failed: " << e.what() << std::endl;
        // 异常会在析构函数中清理资源
    }
    
    delete task_data;

#ifdef DEBUG
    printf("[DEBUG] 🗑️ sync_gpu_task: Task data deleted\n");
#endif
}


void AsyncGpuPipeline::launch_async_pipeline(TaskManager tm, PriorityQueue& q) {

    auto* async_data = new AsyncTaskData(std::move(tm));

    try {
        // 阶段1: 准备数据（同步，但可以优化）
        prepare_gpu_data_stage(*async_data);

        // 阶段2: 异步启动GPU kernel
        launch_kernel_stage(*async_data);

        // 设置kernel完成回调

        // 阶段3: 启动内存拷贝+注册回调（异步）→ 立即返回(这个立即返回埋下了伏笔)
        // start_memory_copy_stage(*async_data);
        // 🔥 修改的阶段3: 把内存拷贝任务提交到线程池
        submit_memory_copy_task(async_data, q);

#ifdef DEBUG
        printf("[DEBUG] ✅ Async pipeline launched successfully\n");
#endif

    } catch (const std::exception& e) {
#ifdef DEBUG
        printf("[DEBUG] ❌ Failed to launch async pipeline: %s\n", e.what());
#endif
        std::cerr << "Failed to launch async pipeline: " << e.what() << std::endl;
        delete async_data;
    }
}



// 🔥 新函数：提交内存拷贝任务到线程池
void submit_memory_copy_task(AsyncGpuPipeline::AsyncTaskData* data, PriorityQueue& q) {
#ifdef DEBUG
    printf("[DEBUG] 📤 submit_memory_copy_task: Submitting memory copy task to thread pool\n");
#endif

#ifdef DEBUG
            printf("[DEBUG] ✅ start launch async  copying results back\n");
#endif
    // 🔥 关键：使用同一个 compute_stream，自动等待kernel完成
    // 这里异步一下， 然后 提交的任务也是异步， 然后接收任务的 线程就需要等待了。
    cudaError_t err = cudaMemcpyAsync(data->gpu_buffer, data->d_guess_buffer,
                                     data->result_len * sizeof(char),
                                     cudaMemcpyDeviceToHost, data->compute_stream);
    if (err != cudaSuccess) {
        std::cerr << "Memory copy scheduling failed: " << cudaGetErrorString(err) << std::endl;

        return;
    }

#ifdef DEBUG
            printf("[DEBUG] ✅ launch async  copying results back succesfully\n");
#endif
    // 🔥 关键：提交到线程池，受到MAX_PENDING_TASKS限制
    thread_pool->enqueue([data, &q]() {
        try {
#ifdef DEBUG
            printf("[DEBUG] 🧵 Memory copy task started - waiting for GPU...\n");
#endif
            
            // 🔥 等待GPU kernel完成（同步点）
            cudaError_t sync_err = cudaStreamSynchronize(data->compute_stream);
            if (sync_err != cudaSuccess) {
                throw std::runtime_error("GPU kernel execution failed: " + std::string(cudaGetErrorString(sync_err)));
            }

#ifdef DEBUG
            printf("[DEBUG] ✅ Memory copy completed, processing results\n");
#endif
            

            
            // CPU处理阶段
            process_strings_stage(*data);
            
            // 合并结果
            merge_results_stage(*data);
            
            // 清理资源
            synchronous_cleanup(*data);

#ifdef DEBUG
            printf("[DEBUG] ✅ Memory copy task completed successfully\n");
#endif

        } catch (const std::exception& e) {
            std::cerr << "Memory copy task failed: " << e.what() << std::endl;
            data->has_error = true;
            
            // 错误时也要清理
            try {
                synchronous_cleanup(*data);
            } catch (...) {
                std::cerr << "Cleanup also failed during error handling" << std::endl;
            }
        }
        
        // 🔥 在任务真正完成后递减计数器
#ifdef TASK_COUNT
        int cur_task = --pending_task_count;
        cout << "now -1 has  " << cur_task << " tasks (memory copy completed)\n";
#endif
    });
#ifdef TASK_COUNT
        int cur_task = ++pending_task_count;
        cout << "now -1 has  " << cur_task << " tasks (memory copy completed)\n";
#endif
#ifdef DEBUG
    printf("[DEBUG] 📤 Memory copy task submitted to thread pool\n");
#endif
}

// 准备GPU数据阶段 (同步, 除开 内存复制)
void prepare_gpu_data_stage(AsyncGpuPipeline::AsyncTaskData& data) {
#ifdef DEBUG
    printf("[DEBUG] 📋 prepare_gpu_data_stage: Starting data preparation...\n");
#endif

    TaskManager& tm = data.task_manager;

    //1. 准备数据（类似原来的逻辑）
    Taskcontent h_tasks;

    data.result_len = 0;

    h_tasks.seg_types = tm.seg_types.data();
    h_tasks.seg_ids = tm.seg_ids.data();

    // ⚠️ 修复：保存字符串到data中
    data.all_prefixes = std::accumulate(tm.prefixs.begin(), tm.prefixs.end(), std::string(""));
    h_tasks.prefixs = data.all_prefixes.c_str();

#ifdef DEBUG
    printf("[DEBUG] 📋 Data preparation: taskcount=%d, guesscount=%d, prefixes_len=%zu\n",
           tm.taskcount, tm.guesscount, data.all_prefixes.length());
#endif

    data.h_prefix_offsets = new int[tm.prefixs.size() + 1];
    h_tasks.prefix_offsets = data.h_prefix_offsets;
    h_tasks.prefix_offsets[0] = 0;

    for (size_t i = 0; i < tm.prefixs.size(); ++i) {
        h_tasks.prefix_offsets[i + 1] = h_tasks.prefix_offsets[i] + tm.prefix_lens[i];
    }

    h_tasks.prefix_lens = tm.prefix_lens.data();
    h_tasks.taskcount = tm.taskcount;
    h_tasks.guesscount = tm.guesscount;
    h_tasks.seg_lens = tm.seg_lens.data();
    h_tasks.seg_value_counts = tm.seg_value_count.data();


    // ⚠️ 修复：保存偏移量到data中
    data.res_offset.clear();  // 清空而不是resize
    data.cumulative_offsets.resize(tm.taskcount + 1, 0);

    // 计算偏移量
    for(int i = 0; i < tm.taskcount; i++){
        data.res_offset.push_back(data.result_len);  // 这样就正确了
        data.result_len += tm.seg_value_count[i] * (tm.seg_lens[i] + tm.prefix_lens[i]);
        data.cumulative_offsets[i + 1] = data.cumulative_offsets[i] + tm.seg_value_count[i];
    }
    h_tasks.output_offsets = data.res_offset.data();
    h_tasks.cumulative_guess_offsets = data.cumulative_offsets.data();

    // 分配host buffer
    data.gpu_buffer = new char[data.result_len];

    //2. 分配GPU内存
    CUDA_CHECK(cudaMalloc(&data.temp_prefixs, h_tasks.prefix_offsets[tm.prefixs.size()] * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&data.d_seg_types, tm.taskcount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&data.d_seg_ids, tm.taskcount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&data.d_seg_lens, tm.seg_lens.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&data.d_prefix_offsets, (tm.prefixs.size() + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&data.d_prefix_lens, tm.prefix_lens.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&data.d_seg_value_counts, tm.seg_value_count.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&data.d_cumulative_guess_offsets, (tm.taskcount + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&data.d_output_offsets, (tm.taskcount + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&data.d_tasks, sizeof(Taskcontent)));
    CUDA_CHECK(cudaMalloc(&data.d_guess_buffer, data.result_len * sizeof(char)));


    
    //3. 🔥 同步拷贝数据到GPU
    CUDA_CHECK(cudaMemcpy(data.d_seg_types, h_tasks.seg_types, tm.taskcount * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data.d_seg_ids, h_tasks.seg_ids, tm.taskcount * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data.d_seg_lens, h_tasks.seg_lens, tm.seg_lens.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data.temp_prefixs, h_tasks.prefixs, h_tasks.prefix_offsets[tm.prefixs.size()] * sizeof(char),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data.d_prefix_offsets, h_tasks.prefix_offsets, (tm.prefixs.size() + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data.d_prefix_lens, h_tasks.prefix_lens, tm.prefixs.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data.d_seg_value_counts, h_tasks.seg_value_counts, tm.seg_value_count.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data.d_cumulative_guess_offsets, h_tasks.cumulative_guess_offsets, (tm.taskcount + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data.d_output_offsets, h_tasks.output_offsets, (tm.taskcount + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));



    // 准备Taskcontent结构
    data.task_content.seg_types = data.d_seg_types;
    data.task_content.seg_ids = data.d_seg_ids;
    data.task_content.seg_lens = data.d_seg_lens;
    data.task_content.prefixs = data.temp_prefixs;
    data.task_content.prefix_offsets = data.d_prefix_offsets;
    data.task_content.prefix_lens = data.d_prefix_lens;
    data.task_content.seg_value_counts = data.d_seg_value_counts;
    data.task_content.cumulative_guess_offsets = data.d_cumulative_guess_offsets;
    data.task_content.output_offsets = data.d_output_offsets;
    data.task_content.taskcount = tm.taskcount;
    data.task_content.guesscount = tm.guesscount;

    CUDA_CHECK(cudaMemcpy(data.d_tasks, &data.task_content, sizeof(Taskcontent),
                          cudaMemcpyHostToDevice));

#ifdef DEBUG
    printf("[DEBUG] 📋 prepare_gpu_data_stage: Data preparation completed, result_len=%zu\n", data.result_len);
#endif
}

// 启动kernel阶段（纯异步）
void launch_kernel_stage(AsyncGpuPipeline::AsyncTaskData& data) {
#ifdef DEBUG
    printf("[DEBUG] 🔥 launch_kernel_stage: Starting kernel launch...\n");
#endif

    // 🔥 第一步：检查所有关键变量是否存在
    printf("[DEBUG] 🔍 Checking all variables before kernel launch:\n");

    // 检查 gpu_data（全局变量）
    if (gpu_data == nullptr) {
        printf("[ERROR] ❌ gpu_data is NULL! This is likely the cause of 'invalid resource handle'\n");
        throw std::runtime_error("gpu_data is not initialized");
    } else {
        printf("[DEBUG] yes gpu_data is valid: %p\n", gpu_data);
    }

    // 检查 data 的关键GPU指针
    printf("[DEBUG] 🔍 Checking AsyncTaskData GPU pointers:\n");
    
    if (data.d_tasks == nullptr) {
        printf("[ERROR] ❌ data.d_tasks is NULL!\n");
        throw std::runtime_error("d_tasks is not allocated");
    } else {
        printf("[DEBUG] yes data.d_tasks is valid: %p\n", data.d_tasks);
    }

    if (data.d_guess_buffer == nullptr) {
        printf("[ERROR] ❌ data.d_guess_buffer is NULL!\n");
        throw std::runtime_error("d_guess_buffer is not allocated");
    } else {
        printf("[DEBUG] yes data.d_guess_buffer is valid: %p\n", data.d_guess_buffer);
    }

    if (data.compute_stream == nullptr) {
        printf("[ERROR] ❌ data.compute_stream is NULL!\n");
        throw std::runtime_error("compute_stream is not created");
    } else {
        printf("[DEBUG] yes data.compute_stream is valid: %p\n", data.compute_stream);
    }

    // 检查其他关键GPU指针
    printf("[DEBUG] 🔍 Checking other GPU memory allocations:\n");
    printf("[DEBUG] - temp_prefixs: %p %s\n", data.temp_prefixs, data.temp_prefixs ? "yes" : "no");
    printf("[DEBUG] - d_seg_types: %p %s\n", data.d_seg_types, data.d_seg_types ? "yes" : "no");
    printf("[DEBUG] - d_seg_ids: %p %s\n", data.d_seg_ids, data.d_seg_ids ? "yes" : "no");
    printf("[DEBUG] - d_seg_lens: %p %s\n", data.d_seg_lens, data.d_seg_lens ? "yes" : "no");
    printf("[DEBUG] - d_prefix_offsets: %p %s\n", data.d_prefix_offsets, data.d_prefix_offsets ? "yes" : "no");
    printf("[DEBUG] - d_prefix_lens: %p %s\n", data.d_prefix_lens, data.d_prefix_lens ? "yes" : "no");
    printf("[DEBUG] - d_seg_value_counts: %p %s\n", data.d_seg_value_counts, data.d_seg_value_counts ? "yes" : "no");
    printf("[DEBUG] - d_cumulative_guess_offsets: %p %s\n", data.d_cumulative_guess_offsets, data.d_cumulative_guess_offsets ? "yes" : "no");
    printf("[DEBUG] - d_output_offsets: %p %s\n", data.d_output_offsets, data.d_output_offsets ? "yes" : "no");

    // 检查TaskManager数据的有效性
    TaskManager& tm = data.task_manager;
    printf("[DEBUG] 🔍 Checking TaskManager data:\n");
    printf("[DEBUG] - taskcount: %d\n", tm.taskcount);
    printf("[DEBUG] - guesscount: %d\n", tm.guesscount);
    printf("[DEBUG] - seg_ids.size(): %zu\n", tm.seg_ids.size());
    printf("[DEBUG] - seg_lens.size(): %zu\n", tm.seg_lens.size());
    printf("[DEBUG] - prefixs.size(): %zu\n", tm.prefixs.size());
    printf("[DEBUG] - result_len: %zu\n", data.result_len);

    // 检查内核启动参数的合理性
    int total_threads_needed = (tm.guesscount + GUESS_PER_THREAD - 1) / GUESS_PER_THREAD;
    int threads_per_block = 1024;
    int blocks = (total_threads_needed + threads_per_block - 1) / threads_per_block;

    printf("[DEBUG] 🔍 Checking kernel launch parameters:\n");
    printf("[DEBUG] - GUESS_PER_THREAD: %d\n", GUESS_PER_THREAD);
    printf("[DEBUG] - total_threads_needed: %d\n", total_threads_needed);
    printf("[DEBUG] - threads_per_block: %d\n", threads_per_block);
    printf("[DEBUG] - blocks: %d\n", blocks);



    // 异步启动kernel
    generate_guesses_kernel<<<blocks, threads_per_block, 0, data.compute_stream>>>
        (gpu_data, data.d_tasks, data.d_guess_buffer);



    // 检查启动错误
    CUDA_CHECK(cudaGetLastError());

#ifdef DEBUG
    printf("[DEBUG] 🔥 launch_kernel_stage: Kernel launched successfully\n");
#endif
}




// 处理字符串阶段
void process_strings_stage(AsyncGpuPipeline::AsyncTaskData& data) {

#ifdef DEBUG
    printf("[DEBUG] 🔍 process_strings_stage: Entry - thread_id=%zu\n", std::this_thread::get_id());
    fflush(stdout);
#endif
    TaskManager& tm = data.task_manager;

    // ⚠️ 检查错误状态
    if (data.has_error) {
#ifdef DEBUG
        printf("[DEBUG] ⚠️ process_strings_stage: Skipping due to error state\n");
        fflush(stdout);
#endif
        return;
    }


#ifdef DEBUG
printf("[DEBUG] 🔍 process_strings_stage: Processing %d segments, %d total guesses\n",
        (int)tm.seg_ids.size(), tm.guesscount);
printf("[DEBUG] 🔍 Buffer info: gpu_buffer=%p, result_len=%zu\n",
        data.gpu_buffer, data.result_len);
fflush(stdout);
#endif

    data.local_guesses.reserve(tm.guesscount);

#ifdef DEBUG
        printf("[DEBUG] 🔍 Reserved space for %d guesses\n", tm.guesscount);
        fflush(stdout);
#endif

    try {
        data.local_guesses.reserve(tm.guesscount);

#ifdef DEBUG
        printf("[DEBUG] 🔍 Reserved space for %d guesses\n", tm.guesscount);
        fflush(stdout);
#endif

        for (int i = 0; i < tm.seg_ids.size(); i++) {
#ifdef DEBUG
            if (i < 3) {  // 只打印前3个，避免输出过多
                printf("[DEBUG] 🔍 Processing segment %d: seg_value_count=%d, seg_len=%d, prefix_len=%d\n",
                       i, tm.seg_value_count[i], tm.seg_lens[i], tm.prefix_lens[i]);
                fflush(stdout);
            }
#endif

            for (int j = 0; j < tm.seg_value_count[i]; j++) {
                int start_offset = data.res_offset[i] + j * (tm.seg_lens[i] + tm.prefix_lens[i]);

#ifdef DEBUG
                if (i < 2 && j < 2) {  // 只打印前几个，避免输出过多
                    printf("[DEBUG] 🔍 Creating guess[%d][%d]: offset=%d, length=%d\n",
                           i, j, start_offset, tm.seg_lens[i] + tm.prefix_lens[i]);
                    fflush(stdout);
                }
#endif

                // 检查边界
                if (start_offset + tm.seg_lens[i] + tm.prefix_lens[i] > data.result_len) {
#ifdef DEBUG
                    printf("[DEBUG] ❌ Buffer overflow! offset=%d, length=%d, result_len=%zu\n",
                           start_offset, tm.seg_lens[i] + tm.prefix_lens[i], data.result_len);
                    fflush(stdout);
#endif
                    data.has_error = true;
                    return;
                }

                data.local_guesses.emplace_back(
                    data.gpu_buffer + start_offset,
                    tm.seg_lens[i] + tm.prefix_lens[i]
                );
            }
        }

#ifdef DEBUG
        printf("[DEBUG] ✅ process_strings_stage: Created %zu guesses successfully\n",
               data.local_guesses.size());
        fflush(stdout);
#endif

    } catch (const std::exception& e) {
#ifdef DEBUG
        printf("[DEBUG] ❌ process_strings_stage exception: %s\n", e.what());
        fflush(stdout);
#endif
        data.has_error = true;
    }

}

// 合并结果阶段
void merge_results_stage(AsyncGpuPipeline::AsyncTaskData& data) {
#ifdef DEBUG
    printf("[DEBUG] 🔗 merge_results_stage: Entry - thread_id=%zu\n", std::this_thread::get_id());
    fflush(stdout);
#endif
    // ⚠️ 检查错误状态
    if (data.has_error) {
#ifdef DEBUG
        printf("[DEBUG] ⚠️ merge_results_stage: Skipping due to error state\n");
        fflush(stdout);
#endif
        return;
    }


#ifdef DEBUG
    printf("[DEBUG] 🔗 merge_results_stage: Merging %zu guesses\n", data.local_guesses.size());
    fflush(stdout);
#endif


    try {
#ifdef DEBUG
        printf("[DEBUG] 🔗 Attempting to acquire locks...\n");
        fflush(stdout);
#endif

        {
            // std::lock_guard<std::mutex> lock1(main_data_mutex);

#ifdef DEBUG
            printf("[DEBUG] 🔗 main_data_mutex acquired\n");
            fflush(stdout);
#endif

            // std::lock_guard<std::mutex> lock2(gpu_buffer_mutex);
            std::scoped_lock lock(main_data_mutex, gpu_buffer_mutex);

#ifdef DEBUG
            printf("[DEBUG] 🔗 gpu_buffer_mutex acquired\n");
            printf("[DEBUG] 🔗 Current queue size: %zu\n", q.guesses.size());
            fflush(stdout);
#endif

            // 插入猜测结果到主队列
            q.guesses.insert(q.guesses.end(),
                             data.local_guesses.begin(),
                             data.local_guesses.end());

#ifdef DEBUG
            printf("[DEBUG] 🔗 Guesses inserted, new queue size: %zu\n", q.guesses.size());
            fflush(stdout);
#endif

            // 将GPU缓冲区指针加入管理列表
            if (data.gpu_buffer != nullptr) {
                pending_gpu_buffers.push_back(data.gpu_buffer);
                data.gpu_buffer = nullptr;
#ifdef DEBUG
                printf("[DEBUG] 🔗 GPU buffer added to pending list, total pending: %zu\n",
                       pending_gpu_buffers.size());
                fflush(stdout);
#endif
            } else {
                cout << " gpu_buffer ptr WRONG in merge_stage" << endl;
            }

#ifdef DEBUG
            printf("[DEBUG] 🔗 Releasing locks...\n");
            fflush(stdout);
#endif
        }

#ifdef DEBUG
        printf("[DEBUG] ✅ merge_results_stage: Completed successfully\n");
        fflush(stdout);
#endif

    } catch (const std::exception& e) {
#ifdef DEBUG
        printf("[DEBUG] ❌ merge_results_stage exception: %s\n", e.what());
        fflush(stdout);
#endif
        data.has_error = true;
    }
}


// 智能清理阶段（正常时异步，异常时同步）
void cleanup_stage(AsyncGpuPipeline::AsyncTaskData& data) {
#ifdef DEBUG
    printf("[DEBUG] 🧹 cleanup_stage: Starting resource cleanup (error=%s)...\n",
           data.has_error ? "true" : "false");
#endif

    if (data.has_error) {
        // 🚨 异常情况：懒得清理了。
#ifdef DEBUG
        printf("[DEBUG] ⚠️ Error detected in cleanup stage\n");
#endif
    } else {
        // ✅ 正常情况：异步清理，提高性能
#ifdef DEBUG
        printf("[DEBUG] ✅ Normal completion, performing asynchronous cleanup\n");
#endif
        synchronous_cleanup(data);
#ifdef DEBUG
        printf("[DEBUG] ✅ Normal completion,  asynchronous cleanup submitted\n");
#endif
    }
}


// 同步步清理（用于正常情况）
void synchronous_cleanup(AsyncGpuPipeline::AsyncTaskData& data) {
#ifdef DEBUG
    printf("[DEBUG] ⚡ synchronous_cleanup: Scheduling async GPU memory release...\n");
#endif

    // 🔥 使用异步释放，不阻塞当前线程
    // if (data.temp_prefixs) cudaFreeAsync(data.temp_prefixs, data.compute_stream);
    // if (data.d_seg_types) cudaFreeAsync(data.d_seg_types, data.compute_stream);
    // if (data.d_seg_ids) cudaFreeAsync(data.d_seg_ids, data.compute_stream);
    // if (data.d_seg_lens) cudaFreeAsync(data.d_seg_lens, data.compute_stream);
    // if (data.d_prefix_offsets) cudaFreeAsync(data.d_prefix_offsets, data.compute_stream);
    // if (data.d_prefix_lens) cudaFreeAsync(data.d_prefix_lens, data.compute_stream);
    // if (data.d_seg_value_counts) cudaFreeAsync(data.d_seg_value_counts, data.compute_stream);
    // if (data.d_cumulative_guess_offsets) cudaFreeAsync(data.d_cumulative_guess_offsets, data.compute_stream);
    // if (data.d_output_offsets) cudaFreeAsync(data.d_output_offsets, data.compute_stream);
    // if (data.d_tasks) cudaFreeAsync(data.d_tasks, data.compute_stream);
    // if (data.d_guess_buffer) cudaFreeAsync(data.d_guess_buffer, data.compute_stream);
    // 🔥 同步释放GPU内存
    cudaError_t sync_err = cudaStreamSynchronize(data.compute_stream);
    if (sync_err != cudaSuccess) {
        std::cerr << "Stream synchronization failed: " << cudaGetErrorString(sync_err) << std::endl;
    }
#ifdef DEBUG
    printf("[DEBUG] ⚡ Stream synchronized, GPU operations completed\n");
#endif
    if (data.temp_prefixs)  cudaFree(data.temp_prefixs);
    if (data.d_seg_types) cudaFree(data.d_seg_types);
    if (data.d_seg_ids) cudaFree(data.d_seg_ids);
    if (data.d_seg_lens) cudaFree(data.d_seg_lens);
    if (data.d_prefix_offsets) cudaFree(data.d_prefix_offsets);
    if (data.d_prefix_lens) cudaFree(data.d_prefix_lens);
    if (data.d_seg_value_counts) cudaFree(data.d_seg_value_counts);
    if (data.d_cumulative_guess_offsets) cudaFree(data.d_cumulative_guess_offsets);
    if (data.d_output_offsets) cudaFree(data.d_output_offsets);
    if (data.d_tasks) cudaFree(data.d_tasks);
    if (data.d_guess_buffer) cudaFree(data.d_guess_buffer);
    data.temp_prefixs = nullptr;
    data.d_seg_types = nullptr;
    data.d_seg_ids = nullptr;
    data.d_seg_lens = nullptr;
    data.d_prefix_offsets = nullptr;
    data.d_prefix_lens = nullptr;
    data.d_seg_value_counts = nullptr;
    data.d_cumulative_guess_offsets = nullptr;
    data.d_output_offsets = nullptr;
    data.d_tasks = nullptr;
    data.d_guess_buffer = nullptr;

    if (data.h_prefix_offsets) {
        delete[] data.h_prefix_offsets;
        data.h_prefix_offsets = nullptr;
    }
#ifdef DEBUG
    printf("[DEBUG] ✅ synchronous_cleanup: All resources cleaned\n");
#endif

#ifdef DEBUG
    cout << "begin delete asynctaskdata" << endl;
#endif
    //改成同步
    delete &data;


#ifdef DEBUG
    cout << "end delete asynctaskdata " << endl;

    if (data.gpu_buffer != nullptr) {
        cout << "Error, gpu_buffer ptr should be nullptr in async_cleanup" << endl;
    }
#endif


}



// 🔥 分阶段GPU任务处理
void process_staged_gpu_task(StagedGpuTask* task, PriorityQueue& q) {
    try {
        switch (task->current_stage) {
            case GpuTaskStage::PREPARE_DATA:
                stage_prepare_data(task, q);
                break;
                
            case GpuTaskStage::LAUNCH_KERNEL:
                stage_launch_kernel(task, q);
                break;
                
            case GpuTaskStage::COPY_RESULTS:
                stage_copy_results(task, q);
                break;
                
            case GpuTaskStage::PROCESS_STRINGS:
                stage_process_strings(task, q);
                break;
                
            case GpuTaskStage::MERGE_RESULTS:
                stage_merge_results(task, q);
                break;
        }
    } catch (const std::exception& e) {
        std::cerr << "Staged GPU task error at stage " << (int)task->current_stage 
                  << ": " << e.what() << std::endl;
        task->has_error = true;
        
        // 错误时直接跳到清理
        cleanup_staged_task(task);
    }
}

// 🔥 阶段1：准备数据并拷贝到GPU
void stage_prepare_data(StagedGpuTask* task, PriorityQueue& q) {
#ifdef DEBUG
    printf("[DEBUG] 📋 Stage 1: Preparing data for GPU...\n");
#endif

    try {
        TaskManager& tm = task->task_manager;

        //1. 准备数据（移植自 TaskManager::launch_gpu_kernel）
        Taskcontent h_tasks;
        task->result_len = 0;

        h_tasks.seg_types = tm.seg_types.data();
        h_tasks.seg_ids = tm.seg_ids.data();

        // 保存所有前缀字符串到task中，确保生命周期
        task->all_prefixes = std::accumulate(tm.prefixs.begin(), tm.prefixs.end(), std::string(""));
        h_tasks.prefixs = task->all_prefixes.c_str();

#ifdef DEBUG
        printf("[DEBUG] 📋 Data preparation: taskcount=%d, guesscount=%d, prefixes_len=%zu\n",
               tm.taskcount, tm.guesscount, task->all_prefixes.length());
        cout << "\nConcatenated prefixes: '" << task->all_prefixes << "'" << endl;
        cout << "Concatenated prefixes length: " << task->all_prefixes.length() << endl;
#endif

        // 计算前缀偏移数组
        task->h_prefix_offsets = new int[tm.prefixs.size() + 1];
        h_tasks.prefix_offsets = task->h_prefix_offsets;
        h_tasks.prefix_offsets[0] = 0;

        for (size_t i = 0; i < tm.prefixs.size(); ++i) {
            h_tasks.prefix_offsets[i + 1] = h_tasks.prefix_offsets[i] + tm.prefix_lens[i];
        }

#ifdef DEBUG
        // cout << "\nPrefix offset info:" << endl;
        // for (size_t i = 0; i <= tm.prefixs.size(); i++) {
        //     cout << "  prefix_offsets[" << i << "] = " << h_tasks.prefix_offsets[i] << endl;
        // }
        
        // cout << "\nVerify prefix extraction:" << endl;
        // for (size_t i = 0; i < tm.prefixs.size(); i++) {
        //     int start = h_tasks.prefix_offsets[i];
        //     int len = tm.prefix_lens[i];
        //     string extracted_prefix(task->all_prefixes.substr(start, len));
        //     cout << "  Task " << i << ": original='" << tm.prefixs[i] 
        //          << "', extracted='" << extracted_prefix << "'" 
        //          << " " << (tm.prefixs[i] == extracted_prefix ? "RIGHT" : "WRONG") << endl;
        // }
#endif

        h_tasks.prefix_lens = tm.prefix_lens.data();
        h_tasks.taskcount = tm.taskcount;
        h_tasks.guesscount = tm.guesscount;
        h_tasks.seg_lens = tm.seg_lens.data();
        h_tasks.seg_value_counts = tm.seg_value_count.data();

        // 🔥 添加数据验证
        if (tm.taskcount <= 0 || tm.guesscount <= 0) {
            throw std::runtime_error("Invalid task counts: taskcount=" + std::to_string(tm.taskcount) + 
                                     ", guesscount=" + std::to_string(tm.guesscount));
        }
        
        if (tm.seg_types.size() != tm.taskcount || tm.seg_ids.size() != tm.taskcount || 
            tm.seg_lens.size() != tm.taskcount || tm.prefix_lens.size() != tm.taskcount ||
            tm.seg_value_count.size() != tm.taskcount) {
            throw std::runtime_error("Inconsistent vector sizes in TaskManager");
        }

        // 计算结果偏移量和累积偏移量
        task->res_offset.clear();
        task->cumulative_offsets.resize(tm.taskcount + 1, 0);

        for(int i = 0; i < tm.taskcount; i++){
            task->res_offset.push_back(task->result_len);
            int segment_size = tm.seg_value_count[i] * (tm.seg_lens[i] + tm.prefix_lens[i]);
            
            // 🔥 检查是否会导致整数溢出
            if (segment_size < 0 || task->result_len > SIZE_MAX - segment_size) {
                throw std::runtime_error("Result buffer size overflow");
            }
            
            task->result_len += segment_size;
            task->cumulative_offsets[i + 1] = task->cumulative_offsets[i] + tm.seg_value_count[i];
        }
        h_tasks.output_offsets = task->res_offset.data();
        h_tasks.cumulative_guess_offsets = task->cumulative_offsets.data();

        // 分配host buffer
        task->gpu_buffer = new char[task->result_len];

#ifdef DEBUG
        printf("[DEBUG] 📋 Starting GPU memory allocation (result_len=%zu)...\n", task->result_len);
#endif

        //2. 分配GPU内存
        int prefix_total_len = h_tasks.prefix_offsets[tm.prefixs.size()];
        // 确保至少分配1字节，避免cudaMalloc(0)的问题
        int alloc_prefix_len = (prefix_total_len > 0) ? prefix_total_len : 1;
        CUDA_CHECK(cudaMalloc(&task->temp_prefixs, alloc_prefix_len * sizeof(char)));

        CUDA_CHECK(cudaMalloc(&task->d_seg_types, tm.taskcount * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&task->d_seg_ids, tm.taskcount * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&task->d_seg_lens, tm.seg_lens.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&task->d_prefix_offsets, (tm.prefixs.size() + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&task->d_prefix_lens, tm.prefix_lens.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&task->d_seg_value_counts, tm.seg_value_count.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&task->d_cumulative_guess_offsets, (tm.taskcount + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&task->d_output_offsets, (tm.taskcount + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&task->d_tasks, sizeof(Taskcontent)));
        CUDA_CHECK(cudaMalloc(&task->d_guess_buffer, task->result_len * sizeof(char)));

#ifdef DEBUG
        printf("[DEBUG] 📋 GPU memory allocated successfully, starting data copy...\n");
#endif

        //3. 同步拷贝数据到GPU
        CUDA_CHECK(cudaMemcpy(task->d_seg_types, h_tasks.seg_types, tm.taskcount * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(task->d_seg_ids, h_tasks.seg_ids, tm.taskcount * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(task->d_seg_lens, h_tasks.seg_lens, tm.seg_lens.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

        // 拷贝前缀数据（只有在有数据时才拷贝）
        if (prefix_total_len > 0) {
            CUDA_CHECK(cudaMemcpy(task->temp_prefixs, h_tasks.prefixs, prefix_total_len * sizeof(char), 
                                  cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaMemcpy(task->d_prefix_offsets, h_tasks.prefix_offsets, (tm.prefixs.size() + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(task->d_prefix_lens, h_tasks.prefix_lens, tm.prefix_lens.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(task->d_seg_value_counts, h_tasks.seg_value_counts, tm.seg_value_count.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(task->d_cumulative_guess_offsets, h_tasks.cumulative_guess_offsets, (tm.taskcount + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(task->d_output_offsets, h_tasks.output_offsets, (tm.taskcount + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));

        // 更新GPU任务结构体中的指针
        Taskcontent d_task_content = h_tasks;
        d_task_content.prefixs = task->temp_prefixs;           // 指向GPU内存
        d_task_content.seg_types = task->d_seg_types;
        d_task_content.seg_ids = task->d_seg_ids;
        d_task_content.seg_lens = task->d_seg_lens;
        d_task_content.prefix_offsets = task->d_prefix_offsets;
        d_task_content.prefix_lens = task->d_prefix_lens;
        d_task_content.seg_value_counts = task->d_seg_value_counts;
        d_task_content.cumulative_guess_offsets = task->d_cumulative_guess_offsets;
        d_task_content.output_offsets = task->d_output_offsets;

        // 拷贝任务结构体到GPU
        CUDA_CHECK(cudaMemcpy(task->d_tasks, &d_task_content, sizeof(Taskcontent), cudaMemcpyHostToDevice));

#ifdef DEBUG
        printf("[DEBUG] ✅ Stage 1 completed: Data prepared and copied to GPU\n");
#endif

        // 更新阶段状态
        task->current_stage = GpuTaskStage::LAUNCH_KERNEL;








    } catch (const std::exception& e) {
#ifdef DEBUG
        printf("[DEBUG] ❌ Stage 1 error: %s\n", e.what());
#endif
        task->has_error = true;
        cleanup_staged_task(task);
        return;
    }

#ifdef DEBUG
    printf("[DEBUG] ✅ Stage 1 completed, data ready on GPU\n");
#endif

    // 🔥 进入下一阶段
    task->current_stage = GpuTaskStage::LAUNCH_KERNEL;
    
    // 🔥 提交下一阶段任务到线程池
    thread_pool->enqueue([task, &q]() {
        process_staged_gpu_task(task, q);
    });
}

// 🔥 阶段2：启动GPU kernel（异步）
void stage_launch_kernel(StagedGpuTask* task, PriorityQueue& q) {
#ifdef DEBUG
    printf("[DEBUG] 🚀 Stage 2: Launching GPU kernel...\n");
#endif

    try {
        TaskManager& tm = task->task_manager;
        
        // 计算kernel参数
        int total_threads_needed = (tm.guesscount + GUESS_PER_THREAD - 1) / GUESS_PER_THREAD;
        int threads_per_block = 1024;
        int blocks = (total_threads_needed + threads_per_block - 1) / threads_per_block;

#ifdef DEBUG
        printf("[DEBUG] 🚀 Kernel params: blocks=%d, threads_per_block=%d, total_threads=%d\n",
               blocks, threads_per_block, total_threads_needed);
#endif

        // 🔥 异步启动kernel（立即返回）
        generate_guesses_kernel<<<blocks, threads_per_block, 0, task->compute_stream>>>
            (gpu_data, task->d_tasks, task->d_guess_buffer);
        
        CUDA_CHECK(cudaGetLastError());

#ifdef DEBUG
    printf("[DEBUG] ✅ Stage 2 completed, kernel launched asynchronously\n");
#endif

        // 🔥 进入下一阶段
        task->current_stage = GpuTaskStage::COPY_RESULTS;
        
        // 🔥 提交下一阶段任务到线程池
        thread_pool->enqueue([task, &q]() {
            process_staged_gpu_task(task, q);
        });
        
    } catch (const std::exception& e) {
#ifdef DEBUG
        printf("[DEBUG] ❌ Stage 2 error: %s\n", e.what());
#endif
        task->has_error = true;
        cleanup_staged_task(task);
    }
}

// 🔥 阶段3：等待GPU完成并拷贝结果
void stage_copy_results(StagedGpuTask* task, PriorityQueue& q) {
#ifdef DEBUG
    printf("[DEBUG] 📥 Stage 3: Waiting for GPU and copying results...\n");
#endif

    try {
        // 🔥 同步等待GPU完成
        cudaError_t sync_err = cudaStreamSynchronize(task->compute_stream);
        if (sync_err != cudaSuccess) {
            throw std::runtime_error("GPU kernel execution failed: " + std::string(cudaGetErrorString(sync_err)));
        }

        // 🔥 同步拷贝结果回CPU
        CUDA_CHECK(cudaMemcpy(task->gpu_buffer, task->d_guess_buffer, 
                             task->result_len * sizeof(char), cudaMemcpyDeviceToHost));

#ifdef DEBUG
printf("[DEBUG] ✅ Stage 3 completed, results copied to CPU\n");
#endif

        // 🔥 进入下一阶段
        task->current_stage = GpuTaskStage::PROCESS_STRINGS;
        
        // 🔥 提交下一阶段任务到线程池
        thread_pool->enqueue([task, &q]() {
            process_staged_gpu_task(task, q);
        });
        
    } catch (const std::exception& e) {
#ifdef DEBUG
        printf("[DEBUG] ❌ Stage 3 error: %s\n", e.what());
#endif
        task->has_error = true;
        cleanup_staged_task(task);
    }
}

// 🔥 阶段4：处理字符串
void stage_process_strings(StagedGpuTask* task, PriorityQueue& q) {
#ifdef DEBUG
    printf("[DEBUG] 🔍 Stage 4: Processing strings...\n");
#endif

    try {
        // 处理字符串，创建string_view
        TaskManager& tm = task->task_manager;
        task->local_guesses.reserve(tm.guesscount);
        
#ifdef DEBUG
        printf("[DEBUG] 🔍 Processing %d segments, %d total guesses, result_len=%zu\n",
               (int)tm.seg_ids.size(), tm.guesscount, task->result_len);
#endif
        
        for (int i = 0; i < tm.seg_ids.size(); i++) {
            for (int j = 0; j < tm.seg_value_count[i]; j++) {
                int start_offset = task->res_offset[i] + j * (tm.seg_lens[i] + tm.prefix_lens[i]);
                
                // 边界检查
                if (start_offset + tm.seg_lens[i] + tm.prefix_lens[i] > task->result_len) {
#ifdef DEBUG
                    printf("[DEBUG] ⚠️ Boundary check failed: start_offset=%d, length=%d, result_len=%zu\n",
                           start_offset, tm.seg_lens[i] + tm.prefix_lens[i], task->result_len);
#endif
                    task->has_error = true;
                    return;
                }
                
                // 创建string_view
                std::string_view guess(
                    task->gpu_buffer + start_offset,
                    tm.seg_lens[i] + tm.prefix_lens[i]
                );
                task->local_guesses.emplace_back(guess);
            }
        }

#ifdef DEBUG
        printf("[DEBUG] ✅ Stage 4 completed: %zu strings processed\n", task->local_guesses.size());
#endif

        // 🔥 进入下一阶段
        task->current_stage = GpuTaskStage::MERGE_RESULTS;
        
        // 🔥 提交下一阶段任务到线程池
        thread_pool->enqueue([task, &q]() {
            process_staged_gpu_task(task, q);
        });
        
    } catch (const std::exception& e) {
#ifdef DEBUG
        printf("[DEBUG] ❌ Stage 4 error: %s\n", e.what());
#endif
        task->has_error = true;
        cleanup_staged_task(task);
    }
}

// 🔥 阶段5：合并结果到全局队列
void stage_merge_results(StagedGpuTask* task, PriorityQueue& q) {
#ifdef DEBUG
    printf("[DEBUG] 🔗 Stage 5: Merging results to global queue...\n");
#endif

    try {
        // 🔥 获取全局锁，合并结果
        {
            std::scoped_lock lock(main_data_mutex, gpu_buffer_mutex);

#ifdef DEBUG
            printf("[DEBUG] 🔗 Locks acquired, current queue size: %zu\n", q.guesses.size());
#endif

            // 插入猜测结果到主队列
            q.guesses.insert(q.guesses.end(), 
                           task->local_guesses.begin(), 
                           task->local_guesses.end());

#ifdef DEBUG
            printf("[DEBUG] 🔗 Guesses inserted, new queue size: %zu\n", q.guesses.size());
#endif

            // 将GPU缓冲区指针加入管理列表
            if (task->gpu_buffer != nullptr) {
                pending_gpu_buffers.push_back(task->gpu_buffer);
                task->gpu_buffer = nullptr;  // 转移所有权给管理器
                
#ifdef DEBUG
                printf("[DEBUG] 🔗 GPU buffer added to pending list, total pending: %zu\n",
                       pending_gpu_buffers.size());
#endif
            }
        }

#ifdef DEBUG
        printf("[DEBUG] ✅ Stage 5 completed: Results merged successfully\n");
#endif

        // 🔥 清理GPU资源
        cleanup_staged_task(task);
        
    } catch (const std::exception& e) {
#ifdef DEBUG
        printf("[DEBUG] ❌ Stage 5 error: %s\n", e.what());
#endif
        task->has_error = true;
        cleanup_staged_task(task);
    }
}

// 🔥 清理阶段任务
void cleanup_staged_task(StagedGpuTask* task) {
#ifdef DEBUG
    printf("[DEBUG] 🧹 Cleaning up staged task (error=%s)...\n",
           task->has_error ? "true" : "false");
#endif

    // 清理GPU资源
    task->cleanup_gpu_resources();
    
    // 删除任务对象
    delete task;

#ifdef DEBUG
    printf("[DEBUG] ✅ Staged task cleanup completed\n");
#endif
}



// // 🔥 阶段5：合并结果并清理
// void stage_merge_results(StagedGpuTask* task, PriorityQueue& q) {
// #ifdef DEBUG
//     printf("[DEBUG] 🔗 Stage 5: Merging results...\n");
// #endif

//     // 合并到全局队列
//     {
//         std::scoped_lock lock(main_data_mutex, gpu_buffer_mutex);
        
//         q.guesses.insert(q.guesses.end(),
//                          task->local_guesses.begin(),
//                          task->local_guesses.end());
        
//         // 转移GPU缓冲区所有权
//         if (task->gpu_buffer != nullptr) {
//             pending_gpu_buffers.push_back(task->gpu_buffer);
//             task->gpu_buffer = nullptr;
//         }
//     }

// #ifdef DEBUG
//     printf("[DEBUG] ✅ Stage 5 completed, %zu guesses merged\n", task->local_guesses.size());
// #endif

//     // 🔥 清理任务
//     cleanup_staged_task(task);
// }

// // 🔥 清理分阶段任务
// void cleanup_staged_task(StagedGpuTask* task) {
// #ifdef DEBUG
//     printf("[DEBUG] 🗑️ Cleaning up staged task...\n");
// #endif

//     // 清理GPU资源
//     task->cleanup_gpu_resources();
    
//     // 删除任务对象
//     delete task;

// #ifdef DEBUG
//     printf("[DEBUG] ✅ Staged task cleanup completed\n");
// #endif
// }

// 🔥 StagedGpuTask的GPU资源清理方法实现
void StagedGpuTask::cleanup_gpu_resources() {
#ifdef DEBUG
    printf("[DEBUG] 🗑️ Cleaning up GPU resources...\n");
#endif

    // 同步等待流完成
    if (compute_stream) {
        cudaError_t sync_err = cudaStreamSynchronize(compute_stream);
        if (sync_err != cudaSuccess) {
            std::cerr << "Stream synchronization failed during cleanup: " 
                      << cudaGetErrorString(sync_err) << std::endl;
        }
    }

    // 释放GPU内存
    if (temp_prefixs) { cudaFree(temp_prefixs); temp_prefixs = nullptr; }
    if (d_seg_types) { cudaFree(d_seg_types); d_seg_types = nullptr; }
    if (d_seg_ids) { cudaFree(d_seg_ids); d_seg_ids = nullptr; }
    if (d_seg_lens) { cudaFree(d_seg_lens); d_seg_lens = nullptr; }
    if (d_prefix_offsets) { cudaFree(d_prefix_offsets); d_prefix_offsets = nullptr; }
    if (d_prefix_lens) { cudaFree(d_prefix_lens); d_prefix_lens = nullptr; }
    if (d_seg_value_counts) { cudaFree(d_seg_value_counts); d_seg_value_counts = nullptr; }
    if (d_cumulative_guess_offsets) { cudaFree(d_cumulative_guess_offsets); d_cumulative_guess_offsets = nullptr; }
    if (d_output_offsets) { cudaFree(d_output_offsets); d_output_offsets = nullptr; }
    if (d_tasks) { cudaFree(d_tasks); d_tasks = nullptr; }
    if (d_guess_buffer) { cudaFree(d_guess_buffer); d_guess_buffer = nullptr; }

    // 释放CPU内存
    if (h_prefix_offsets) {
        delete[] h_prefix_offsets;
        h_prefix_offsets = nullptr;
    }

#ifdef DEBUG
    printf("[DEBUG] ✅ GPU resources cleaned successfully\n");
#endif
}

