#include "guessing_cuda.h"
#include "PCFG.h"


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            fprintf(stderr, "\033[1;31m[CUDA ERROR]\033[0m %s:%d: %s (%d)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err__), err__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

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





void AsyncGpuPipeline::launch_async_pipeline(TaskManager tm, PriorityQueue& q) {
    
    auto* async_data = new AsyncTaskData(std::move(tm));
    
    try {
        // 阶段1: 准备数据（同步，但可以优化）
        prepare_gpu_data_stage(*async_data);
        
        // 阶段2: 异步启动GPU kernel
        launch_kernel_stage(*async_data);
        
        // 设置kernel完成回调
   
        // 阶段3: 启动内存拷贝+注册回调（异步）→ 立即返回
        start_memory_copy_stage(*async_data);

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

// 准备GPU数据阶段 (同步, 除开 内存复制)
void prepare_gpu_data_stage(AsyncGpuPipeline::AsyncTaskData& data) {
#ifdef DEBUG
    printf("[DEBUG] 📋 prepare_gpu_data_stage: Starting data preparation...\n");
#endif
    
    TaskManager& tm = data.task_manager;
    
    //1. 准备数据（类似原来的逻辑）
    Taskcontent h_tasks; 
    Taskcontent temp;
    
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
    
    //3. 异步拷贝数据到GPU
    CUDA_CHECK(cudaMemcpyAsync(data.d_seg_types, h_tasks.seg_types, tm.taskcount * sizeof(int), 
                              cudaMemcpyHostToDevice, data.compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(data.d_seg_ids, h_tasks.seg_ids, tm.taskcount * sizeof(int), 
                              cudaMemcpyHostToDevice, data.compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(data.d_seg_lens, h_tasks.seg_lens, tm.seg_lens.size() * sizeof(int), 
                              cudaMemcpyHostToDevice, data.compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(data.temp_prefixs, h_tasks.prefixs, h_tasks.prefix_offsets[tm.prefixs.size()] * sizeof(char), 
                              cudaMemcpyHostToDevice, data.compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(data.d_prefix_offsets, h_tasks.prefix_offsets, (tm.prefixs.size() + 1) * sizeof(int), 
                              cudaMemcpyHostToDevice, data.compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(data.d_prefix_lens, h_tasks.prefix_lens, tm.prefixs.size() * sizeof(int), 
                              cudaMemcpyHostToDevice, data.compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(data.d_seg_value_counts, h_tasks.seg_value_counts, tm.seg_value_count.size() * sizeof(int), 
                              cudaMemcpyHostToDevice, data.compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(data.d_cumulative_guess_offsets, h_tasks.cumulative_guess_offsets, (tm.taskcount + 1) * sizeof(int), 
                              cudaMemcpyHostToDevice, data.compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(data.d_output_offsets, h_tasks.output_offsets, (tm.taskcount + 1) * sizeof(int), 
                              cudaMemcpyHostToDevice, data.compute_stream));
    
    // 准备Taskcontent结构
    temp.seg_types = data.d_seg_types;
    temp.seg_ids = data.d_seg_ids;
    temp.seg_lens = data.d_seg_lens;
    temp.prefixs = data.temp_prefixs;
    temp.prefix_offsets = data.d_prefix_offsets;
    temp.prefix_lens = data.d_prefix_lens;
    temp.seg_value_counts = data.d_seg_value_counts;
    temp.cumulative_guess_offsets = data.d_cumulative_guess_offsets;
    temp.output_offsets = data.d_output_offsets;
    temp.taskcount = tm.taskcount;
    temp.guesscount = tm.guesscount;
    
    CUDA_CHECK(cudaMemcpyAsync(data.d_tasks, &temp, sizeof(Taskcontent), 
                              cudaMemcpyHostToDevice, data.compute_stream));
#ifdef DEBUG
    printf("[DEBUG] 📋 prepare_gpu_data_stage: Data preparation completed, result_len=%zu\n", data.result_len);
#endif
}

// 启动kernel阶段（纯异步）
void launch_kernel_stage(AsyncGpuPipeline::AsyncTaskData& data) {
#ifdef DEBUG
    printf("[DEBUG] 🔥 launch_kernel_stage: Starting kernel launch...\n");
#endif    
    TaskManager& tm = data.task_manager;
    
    int total_threads_needed = (tm.guesscount + GUESS_PER_THREAD - 1) / GUESS_PER_THREAD;
    int threads_per_block = 1024;
    int blocks = (total_threads_needed + threads_per_block - 1) / threads_per_block;

#ifdef DEBUG
    printf("[DEBUG] 🔥 Kernel config: guesscount=%d, blocks=%d, threads_per_block=%d\n", 
           tm.guesscount, blocks, threads_per_block);
#endif
    
    // 异步启动kernel
    generate_guesses_kernel<<<blocks, threads_per_block, 0, data.compute_stream>>>
        (gpu_data, data.d_tasks, data.d_guess_buffer);
        

    
    // 检查启动错误
    CUDA_CHECK(cudaGetLastError());

#ifdef DEBUG
    printf("[DEBUG] 🔥 launch_kernel_stage: Kernel launched successfully\n");
#endif
}


// 启动内存拷贝阶段（完全异步）
void start_memory_copy_stage(AsyncGpuPipeline::AsyncTaskData& data) {
#ifdef DEBUG
    printf("[DEBUG] 📥 start_memory_copy_stage: Scheduling async memory copy...\n");
#endif

    // 🔥 关键：使用同一个 compute_stream，自动等待kernel完成
    cudaError_t err = cudaMemcpyAsync(data.gpu_buffer, data.d_guess_buffer, 
                                     data.result_len * sizeof(char), 
                                     cudaMemcpyDeviceToHost, data.compute_stream);
    if (err != cudaSuccess) {
        std::cerr << "Memory copy scheduling failed: " << cudaGetErrorString(err) << std::endl;

        return;
    }

    // 🔥 关键：注册内存拷贝完成回调
    err = cudaStreamAddCallback(data.compute_stream, 
                               AsyncGpuPipeline::memory_copy_completion_callback, 
                               &data, 0);
    if (err != cudaSuccess) {
        std::cerr << "Callback registration failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

#ifdef DEBUG
    printf("[DEBUG] 📥 Memory copy scheduled with callback, stage 3 returning immediately\n");
#endif

    // 🔥 关键：立即返回！后续处理在回调中完成
}



// 内存拷贝完成回调 - 打包任务给线程池
void CUDART_CB AsyncGpuPipeline::memory_copy_completion_callback(cudaStream_t stream, cudaError_t status, void* userData) {
#ifdef DEBUG
    printf("[DEBUG] 📥 memory_copy_completion_callback: Entry - userData=%p, status=%d\n", 
           userData, status);
    fflush(stdout);
#endif

    if (!userData) {
#ifdef DEBUG
        printf("[DEBUG] ❌ userData is null in callback!\n");
        fflush(stdout);
#endif
        return;
    }

    auto* data = static_cast<AsyncGpuPipeline::AsyncTaskData*>(userData);
    

    //BUG 清理函数感觉不太对
    if (status != cudaSuccess) {
        std::cerr << "Memory copy failed: " << cudaGetErrorString(status) << std::endl;
        data->has_error = true;
        // 错误就别管了。

        return;
    }

#ifdef DEBUG
    printf("[DEBUG] ✅ Memory copy successful, submitting CPU processing task to thread pool\n");
#endif

    // 🔥 关键：打包CPU处理任务，提交给线程池
    thread_pool->enqueue([data]() {
        try {
#ifdef DEBUG
            printf("[DEBUG] 🧵 Thread pool task started - processing results\n");
#endif
            // CPU处理阶段
            process_strings_stage(*data);

#ifdef DEBUG
            printf("[DEBUG] ✅ process_strings_stage completed s\n");
            fflush(stdout);
#endif
            merge_results_stage(*data);
#ifdef DEBUG
            printf("[DEBUG] ✅ merge_results_stage completed inms\n");
            fflush(stdout);
#endif
            cleanup_stage(*data);
            
#ifdef DEBUG
            printf("[DEBUG] ✅ Thread pool task completed successfully\n");
#endif
            
        } catch (const std::exception& e) {
            std::cerr << "CPU processing failed: " << e.what() << std::endl;
            data->has_error = true;
            // 错误就别管了。
            // cleanup_stage(*data);
        }
        pending_task_count--;
        int cur_task = pending_task_count.load();
        cout << "now -1 has  " << cur_task << " tasks\n";

    });
    pending_task_count++;
    int cur_task = pending_task_count.load();
    cout << "now +1 has  " << cur_task << " tasks\n";

#ifdef DEBUG
    printf("[DEBUG] 📤 CPU processing task submitted, callback returning\n");
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
            std::lock_guard<std::mutex> lock1(main_data_mutex);
            
#ifdef DEBUG
            printf("[DEBUG] 🔗 main_data_mutex acquired\n");
            fflush(stdout);
#endif
            
            std::lock_guard<std::mutex> lock2(gpu_buffer_mutex);
            
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
        asynchronous_cleanup(data);
#ifdef DEBUG
        printf("[DEBUG] ✅ Normal completion,  asynchronous cleanup submitted\n");
#endif
    }
}


// 异步清理（用于正常情况）
void asynchronous_cleanup(AsyncGpuPipeline::AsyncTaskData& data) {
#ifdef DEBUG
    printf("[DEBUG] ⚡ asynchronous_cleanup: Scheduling async GPU memory release...\n");
#endif

    // 🔥 使用异步释放，不阻塞当前线程
    if (data.temp_prefixs) cudaFreeAsync(data.temp_prefixs, data.compute_stream);
    if (data.d_seg_types) cudaFreeAsync(data.d_seg_types, data.compute_stream);
    if (data.d_seg_ids) cudaFreeAsync(data.d_seg_ids, data.compute_stream);
    if (data.d_seg_lens) cudaFreeAsync(data.d_seg_lens, data.compute_stream);
    if (data.d_prefix_offsets) cudaFreeAsync(data.d_prefix_offsets, data.compute_stream);
    if (data.d_prefix_lens) cudaFreeAsync(data.d_prefix_lens, data.compute_stream);
    if (data.d_seg_value_counts) cudaFreeAsync(data.d_seg_value_counts, data.compute_stream);
    if (data.d_cumulative_guess_offsets) cudaFreeAsync(data.d_cumulative_guess_offsets, data.compute_stream);
    if (data.d_output_offsets) cudaFreeAsync(data.d_output_offsets, data.compute_stream);
    if (data.d_tasks) cudaFreeAsync(data.d_tasks, data.compute_stream);
    if (data.d_guess_buffer) cudaFreeAsync(data.d_guess_buffer, data.compute_stream);


    // 🔥 关键：设置异步清理完成回调
    cudaStreamAddCallback(data.compute_stream, 
                         AsyncGpuPipeline::async_cleanup_completion_callback, 
                         &data, 0);
    // 释放CPU内存
    if (data.h_prefix_offsets) {
        delete[] data.h_prefix_offsets;
        data.h_prefix_offsets = nullptr;
    }

#ifdef DEBUG
    if (data.gpu_buffer != nullptr) {
        cout << "Error, gpu_buffer ptr should be nullptr in async_cleanup" << endl;
    }
#endif
   
#ifdef DEBUG
    printf("[DEBUG] 📥 Async cleanup scheduled with callback, returning immediately\n");
#endif
    
    // 🔥 立即返回，不等待清理完成
}

// 异步清理完成回调
void CUDART_CB AsyncGpuPipeline::async_cleanup_completion_callback(cudaStream_t stream, cudaError_t status, void* userData) {
#ifdef DEBUG
    printf("[DEBUG] 🧹 async_cleanup_completion_callback: GPU memory released\n");
#endif

    auto* data = static_cast<AsyncGpuPipeline::AsyncTaskData*>(userData);
    
    if (status != cudaSuccess) {
        std::cerr << "Async cleanup failed: " << cudaGetErrorString(status) << std::endl;
        // 别清理了， 减少错误。
        return;
    }

#ifdef DEBUG
    printf("[DEBUG] ✅ GPU memory cleanup successful, cleaning CPU memory\n");
#endif

#ifdef DEBUG
    printf("[DEBUG] 🗑️ cleanup_cpu_memory: Releasing CPU memory...\n");
#endif

    // 释放CPU内存
    if (data->h_prefix_offsets) {
        delete[] data->h_prefix_offsets;
        data->h_prefix_offsets = nullptr;
    }
    
#ifdef DEBUG
    printf("[DEBUG] ✅ CPU memory cleanup completed\n");
#endif
    
    delete data;
    
#ifdef DEBUG
    printf("[DEBUG] ✅ AsyncTaskData deleted, pipeline fully completed\n");
#endif


}
